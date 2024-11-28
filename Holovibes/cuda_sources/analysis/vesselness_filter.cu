#include "vesselness_filter.cuh"

#include "cuda_memory.cuh"
#include "tools_analysis.cuh"
#include "tools_analysis_debug.hh"

using holovibes::cuda_tools::CufftHandle;

__global__ void convolution_kernel(float* output,
                                   const float* input,
                                   const float* kernel,
                                   int width,
                                   int height,
                                   int kWidth,
                                   int kHeight,
                                   ConvolutionPaddingType padding_type,
                                   int padding_scalar = 0)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return; // Ensure we don't go out of image bounds

    float result = 0.0f;
    int kHalfWidth = kWidth / 2;
    int kHalfHeight = kHeight / 2;

    // Apply the convolution with replicate boundary behavior
    for (int ky = -kHalfHeight; ky <= kHalfHeight; ++ky)
    {
        for (int kx = -kHalfWidth; kx <= kHalfWidth; ++kx)
        {
            // Calculate the coordinates for the image
            int ix = x + kx;
            int iy = y + ky;

            float imageValue;
            // Appliquer le type de padding
            if (padding_type == ConvolutionPaddingType::REPLICATE)
            {
                // Comportement de réplication des bords
                if (ix < 0)
                    ix = 0;
                if (ix >= width)
                    ix = width - 1;
                if (iy < 0)
                    iy = 0;
                if (iy >= height)
                    iy = height - 1;

                imageValue = input[iy * width + ix];
            }
            else if (padding_type == ConvolutionPaddingType::SCALAR)
            {
                // Utiliser la valeur du padding scalar
                if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                    imageValue = padding_scalar;
                else
                    imageValue = input[iy * width + ix];
            }

            float kernelValue = kernel[(ky + kHalfHeight) * kWidth + (kx + kHalfWidth)];
            result += imageValue * kernelValue;
        }
    }

    output[y * width + x] = result;
}

void apply_convolution(float* const input_output,
                       const float* kernel,
                       size_t width,
                       size_t height,
                       size_t kWidth,
                       size_t kHeight,
                       float* const convolution_tmp_buffer,
                       cudaStream_t stream,
                       ConvolutionPaddingType padding_type,
                       int padding_scalar)
{
    // Définir la taille des blocs et de la grille
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Lancer le kernel
    convolution_kernel<<<gridSize, blockSize, 0, stream>>>(convolution_tmp_buffer,
                                                           input_output,
                                                           kernel,
                                                           width,
                                                           height,
                                                           kWidth,
                                                           kHeight,
                                                           padding_type,
                                                           padding_scalar);

    cudaCheckError();

    // Copy convolution result in the input_output
    cudaXMemcpyAsync(input_output,
                     convolution_tmp_buffer,
                     sizeof(float) * width * height,
                     cudaMemcpyDeviceToDevice,
                     stream);
}

void gaussian_imfilter_sep(float* input_output,
                           float* gpu_kernel_buffer,
                           int kernel_x_size,
                           int kernel_y_size,
                           const size_t frame_res,
                           float* const convolution_tmp_buffer,
                           cudaStream_t stream)
{
    // This convolution method gives correct values compared to matlab
    apply_convolution(input_output,
                      gpu_kernel_buffer,
                      std::sqrt(frame_res),
                      std::sqrt(frame_res),
                      kernel_x_size,
                      kernel_y_size,
                      convolution_tmp_buffer,
                      stream,
                      ConvolutionPaddingType::REPLICATE);
}

__global__ void kernel_abs_lambda_division(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] = abs(lambda_1[index]) / abs(lambda_2[index]);
}

void abs_lambda_division(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_abs_lambda_division<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res);
    cudaCheckError();
}

__global__ void kernel_normalize(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] = sqrtf(powf(lambda_1[index], 2) + powf(lambda_2[index], 2));
}

void normalize(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_normalize<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res);
    cudaCheckError();
}

__global__ void kernel_If(float* output, size_t input_size, float* R_blob, float beta, float c, float* c_temp)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        float A = powf(R_blob[index], 2);
        float B = 2 * powf(beta, 2);
        float C = expf(-(A / B));
        float D = 2 * powf(c, 2);
        float E = c_temp[index] / D;
        float F = 1 - expf(-E);
        output[index] = C * F;
    }
}

void If(float* output, size_t input_size, float* R_blob, float beta, float c, float* c_temp, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_If<<<blocks, threads, 0, stream>>>(output, input_size, R_blob, beta, c, c_temp);
    cudaCheckError();
}

__global__ void kernel_lambda_2_logical(float* output, size_t input_size, float* lambda_2)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] *= (lambda_2[index] <= 0.f);
}

void lambda_2_logical(float* output, size_t input_size, float* lambda_2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_lambda_2_logical<<<blocks, threads, 0, stream>>>(output, input_size, lambda_2);
    cudaCheckError();
}

void compute_I(float* output,
               float* input,
               float* g_mul,
               float A,
               uint frame_res,
               uint kernel_x_size,
               uint kernel_y_size,
               float* const convolution_tmp_buffer,
               cudaStream_t stream)
{
    cudaXMemcpyAsync(output, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    gaussian_imfilter_sep(output, g_mul, kernel_x_size, kernel_y_size, frame_res, convolution_tmp_buffer, stream);

    multiply_array_by_scalar(output, frame_res, A, stream);
}

void vesselness_filter(float* output,
                       float* input,
                       float sigma,
                       float* g_xx_mul,
                       float* g_xy_mul,
                       float* g_yy_mul,
                       int kernel_x_size,
                       int kernel_y_size,
                       int frame_res,
                       holovibes::VesselnessFilterStruct& filter_struct_,
                       cublasHandle_t cublas_handler,
                       cudaStream_t stream)
{
    int gamma = 1;
    float beta = 0.8f;

    float A = std::pow(sigma, gamma);

    compute_I(filter_struct_.I,
              input,
              g_xx_mul,
              A,
              frame_res,
              kernel_x_size,
              kernel_y_size,
              filter_struct_.convolution_tmp_buffer,
              stream);

    prepare_hessian(filter_struct_.H, filter_struct_.I, frame_res, 0, stream);

    compute_I(filter_struct_.I,
              input,
              g_xy_mul,
              A,
              frame_res,
              kernel_x_size,
              kernel_y_size,
              filter_struct_.convolution_tmp_buffer,
              stream);

    prepare_hessian(filter_struct_.H, filter_struct_.I, frame_res, 1, stream);

    compute_I(filter_struct_.I,
              input,
              g_yy_mul,
              A,
              frame_res,
              kernel_x_size,
              kernel_y_size,
              filter_struct_.convolution_tmp_buffer,
              stream);

    prepare_hessian(filter_struct_.H, filter_struct_.I, frame_res, 2, stream);

    cudaXMemsetAsync(filter_struct_.lambda_1, 0, frame_res * sizeof(float), stream);
    cudaXMemsetAsync(filter_struct_.lambda_2, 0, frame_res * sizeof(float), stream);

    compute_eigen_values(filter_struct_.H, frame_res, filter_struct_.lambda_1, filter_struct_.lambda_2, stream);

    abs_lambda_division(filter_struct_.R_blob, filter_struct_.lambda_1, filter_struct_.lambda_2, frame_res, stream);
    normalize(filter_struct_.c_temp, filter_struct_.lambda_1, filter_struct_.lambda_2, frame_res, stream);

    int c_index;
    cublasStatus_t status = cublasIsamax(cublas_handler, frame_res, filter_struct_.c_temp, 1, &c_index);

    float c;
    cudaXMemcpyAsync(&c, &filter_struct_.c_temp[c_index - 1], sizeof(float), cudaMemcpyDeviceToHost, stream);

    If(output, frame_res, filter_struct_.R_blob, beta, c, filter_struct_.c_temp, stream);

    lambda_2_logical(output, frame_res, filter_struct_.lambda_2, stream);
}