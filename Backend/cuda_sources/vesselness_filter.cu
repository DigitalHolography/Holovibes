#include "vesselness_filter.cuh"

#include "cuda_memory.cuh"
#include "tools_analysis.cuh"
#include "tools_analysis_debug.hh"
#include "map.cuh"

using holovibes::cuda_tools::CufftHandle;

/*!
 * \brief Performs a 2D convolution operation on an input image using a specified kernel.
 *
 * This CUDA kernel function performs a 2D convolution operation on an input image using a specified kernel.
 * It supports different padding types, including replicate boundary behavior and scalar padding.
 * The result of the convolution is stored in the output array.
 *
 * \param [out] output Pointer to the output array where the convolution result will be stored.
 * \param [in] input Pointer to the input image array.
 * \param [in] kernel Pointer to the convolution kernel array.
 * \param [in] width The width of the input image.
 * \param [in] height The height of the input image.
 * \param [in] kWidth The width of the convolution kernel.
 * \param [in] kHeight The height of the convolution kernel.
 * \param [in] padding_type The type of padding to use (e.g., replicate boundary behavior or scalar padding).
 * \param [in] padding_scalar The scalar value to use for padding if `padding_type` is `SCALAR`. Default is 0.
 *
 * \note The function performs the convolution operation only for the pixels within the specified width and height.
 *       It uses the specified padding type to handle boundary conditions. The kernel is assumed to be centered.
 */
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
        return;

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
            if (padding_type == ConvolutionPaddingType::REPLICATE)
            {
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
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

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

    cudaXMemcpyAsync(input_output,
                     convolution_tmp_buffer,
                     sizeof(float) * width * height,
                     cudaMemcpyDeviceToDevice,
                     stream);
}

/*!
 * \brief Applies a separable Gaussian filter to an input image.
 *
 * This function applies a separable Gaussian filter to an input image using a custom convolution method.
 * It configures and launches a CUDA kernel to perform the convolution operation with replicate padding.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in,out] input_output Pointer to the input-output array where the filtered image will be stored.
 * \param [in] gpu_kernel_buffer Pointer to the GPU buffer containing the separable Gaussian kernel.
 * \param [in] kernel_x_size The width of the Gaussian kernel.
 * \param [in] kernel_y_size The height of the Gaussian kernel.
 * \param [in] frame_res The resolution of the input frame (number of elements per frame).
 * \param [in] convolution_tmp_buffer Pointer to a temporary buffer used to store the intermediate convolution result.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function assumes that the input frame is square, i.e., the width and height are equal and can be derived
 * from the frame resolution. It calls the `apply_convolution` function to perform the convolution operation with
 * replicate padding.
 */
void gaussian_imfilter_sep(float* input_output,
                           float* gpu_kernel_buffer,
                           int kernel_x_size,
                           int kernel_y_size,
                           const size_t frame_res,
                           float* const convolution_tmp_buffer,
                           cudaStream_t stream)
{
    // This convolution method gives same values as matlab
    // We don't use CUDA convolution because it gives us wrong values, maybe we did something wrong
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

/*!
 * \brief Performs element-wise division of the absolute values of two input arrays.
 *
 * This CUDA kernel function performs element-wise division of the absolute values of two input arrays, `lambda_1` and
 * `lambda_2`. The result is stored in the output array. Each thread in the kernel computes the division for a single
 * element.
 *
 * \param [out] output Pointer to the output array where the result of the division will be stored.
 * \param [in] lambda_1 Pointer to the first input array.
 * \param [in] lambda_2 Pointer to the second input array.
 * \param [in] input_size The number of elements in the input arrays.
 *
 * \note The function performs the division only for the elements within the specified input size.
 *       It computes the absolute value of each element before performing the division to ensure non-negative results.
 */
__global__ void kernel_abs_lambda_division(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] = abs(lambda_1[index]) / abs(lambda_2[index]);
}

/*!
 * \brief Performs element-wise division of the absolute values of two input arrays using a CUDA kernel.
 *
 * This function performs element-wise division of the absolute values of two input arrays, `lambda_1` and `lambda_2`,
 * using a CUDA kernel. It configures and launches the kernel to perform the division operation. The function uses the
 * provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the result of the division will be stored.
 * \param [in] lambda_1 Pointer to the first input array.
 * \param [in] lambda_2 Pointer to the second input array.
 * \param [in] frame_res The number of elements in the input arrays.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the frame resolution.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void abs_lambda_division(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_abs_lambda_division<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res);
    cudaCheckError();
}

/*!
 * \brief Computes the element-wise normalization of two input arrays.
 *
 * This CUDA kernel function computes the element-wise normalization of two input arrays, `lambda_1` and `lambda_2`.
 * The normalization is performed by calculating the square root of the sum of the squares of the corresponding elements
 * from the two input arrays. The result is stored in the output array.
 *
 * \param [out] output Pointer to the output array where the result of the normalization will be stored.
 * \param [in] lambda_1 Pointer to the first input array.
 * \param [in] lambda_2 Pointer to the second input array.
 * \param [in] input_size The number of elements in the input arrays.
 *
 * \note The function performs the normalization only for the elements within the specified input size.
 *       It computes the square root of the sum of the squares of the corresponding elements to ensure the
 * normalization.
 */
__global__ void kernel_normalize(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] = sqrtf(powf(lambda_1[index], 2) + powf(lambda_2[index], 2));
}

/*!
 * \brief Computes the element-wise normalization of two input arrays using a CUDA kernel.
 *
 * This function computes the element-wise normalization of two input arrays, `lambda_1` and `lambda_2`,
 * using a CUDA kernel. It configures and launches the kernel to perform the normalization operation.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the result of the normalization will be stored.
 * \param [in] lambda_1 Pointer to the first input array.
 * \param [in] lambda_2 Pointer to the second input array.
 * \param [in] frame_res The number of elements in the input arrays.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the frame resolution.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void normalize(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_normalize<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res);
    cudaCheckError();
}

/*!
 * \brief Computes a custom function for each element in the input arrays.
 *
 * This CUDA kernel function computes a custom function for each element in the input arrays. The function involves
 * several mathematical operations including squaring, exponentiation, and division. The result is stored in the output
 * array.
 *
 * \param [out] output Pointer to the output array where the result will be stored.
 * \param [in] input_size The number of elements in the input arrays.
 * \param [in] R_blob Pointer to the first input array.
 * \param [in] beta A scalar value used in the computation.
 * \param [in] c A scalar value used in the computation.
 * \param [in] c_temp Pointer to the second input array.
 *
 * \note The function performs the computation only for the elements within the specified input size.
 *       It involves several mathematical operations to compute the final result for each element, all coming from
 *       MatLab, Pulsewave project.
 */
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

/*!
 * \brief Computes a custom function for each element in the input arrays using a CUDA kernel.
 *
 * This function computes a custom function for each element in the input arrays, `R_blob` and `c_temp`,
 * using a CUDA kernel, all coming from MatLab, Pulsewave project. It configures and launches the kernel to perform the
 * computation. The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the result will be stored.
 * \param [in] input_size The number of elements in the input arrays.
 * \param [in] R_blob Pointer to the first input array.
 * \param [in] beta A scalar value used in the computation.
 * \param [in] c A scalar value used in the computation.
 * \param [in] c_temp Pointer to the second input array.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the input size.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void If(float* output, size_t input_size, float* R_blob, float beta, float c, float* c_temp, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_If<<<blocks, threads, 0, stream>>>(output, input_size, R_blob, beta, c, c_temp);
    cudaCheckError();
}

/*!
 * \brief Applies a logical condition to elements of an input array.
 *
 * This CUDA kernel function applies a logical condition to elements of an input array. Specifically, it multiplies
 * each element of the output array by 1 if the corresponding element in the `lambda_2` array is less than or equal to
 * 0, and by 0 otherwise. The result is stored in the output array.
 *
 * \param [in,out] output Pointer to the output array where the result will be stored.
 * \param [in] input_size The number of elements in the input arrays.
 * \param [in] lambda_2 Pointer to the input array containing the values to be checked against the logical condition.
 *
 * \note The function performs the logical condition check and multiplication only for the elements within the specified
 * input size.
 */
__global__ void kernel_lambda_2_logical(float* output, size_t input_size, float* lambda_2)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] *= (lambda_2[index] <= 0.f);
}

/*!
 * \brief Applies a logical condition to elements of an input array using a CUDA kernel.
 *
 * This function applies a logical condition to elements of an input array, `lambda_2`, using a CUDA kernel.
 * It configures and launches the kernel to perform the logical condition check and multiplication operation.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [in,out] output Pointer to the output array where the result will be stored.
 * \param [in] input_size The number of elements in the input arrays.
 * \param [in] lambda_2 Pointer to the input array containing the values to be checked against the logical condition.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the input size.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void lambda_2_logical(float* output, size_t input_size, float* lambda_2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_lambda_2_logical<<<blocks, threads, 0, stream>>>(output, input_size, lambda_2);
    cudaCheckError();
}

/*!
 * \brief Computes a processed image by applying a series of operations.
 *
 * This function computes a processed image by applying a series of operations to the input image.
 * It performs the following steps:
 * 1. Copies the input image to the output buffer.
 * 2. Applies a separable Gaussian filter to the output buffer.
 * 3. Multiplies the output buffer by a scalar value.
 * The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the processed image will be stored.
 * \param [in] input Pointer to the input image array.
 * \param [in] g_mul Pointer to the Gaussian kernel array.
 * \param [in] A Scalar value used to multiply the output buffer.
 * \param [in] frame_res The resolution of the input frame (number of elements per frame).
 * \param [in] kernel_x_size The width of the Gaussian kernel.
 * \param [in] kernel_y_size The height of the Gaussian kernel.
 * \param [in] convolution_tmp_buffer Pointer to a temporary buffer used for the convolution operation.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function assumes that the input frame is square, i.e., the width and height are equal and can be derived
 * from the frame resolution. It calls `cudaXMemcpyAsync` to copy the input image to the output buffer,
 * `gaussian_imfilter_sep` to apply the Gaussian filter, and `map_multiply` to multiply the output buffer by the scalar
 * value.
 */
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

    map_multiply(output, frame_res, A, stream);
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
                       holovibes::VesselnessFilterEnv& filter_struct_,
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

    cudaXMemcpyAsync(filter_struct_.H, filter_struct_.I, sizeof(float) * frame_res, cudaMemcpyDeviceToDevice, stream);

    compute_I(filter_struct_.I,
              input,
              g_xy_mul,
              A,
              frame_res,
              kernel_x_size,
              kernel_y_size,
              filter_struct_.convolution_tmp_buffer,
              stream);

    cudaXMemcpyAsync(filter_struct_.H + frame_res,
                     filter_struct_.I,
                     sizeof(float) * frame_res,
                     cudaMemcpyDeviceToDevice,
                     stream);

    compute_I(filter_struct_.I,
              input,
              g_yy_mul,
              A,
              frame_res,
              kernel_x_size,
              kernel_y_size,
              filter_struct_.convolution_tmp_buffer,
              stream);

    cudaXMemcpyAsync(filter_struct_.H + frame_res * 2,
                     filter_struct_.I,
                     sizeof(float) * frame_res,
                     cudaMemcpyDeviceToDevice,
                     stream);

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