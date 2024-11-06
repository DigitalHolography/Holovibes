#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"

__global__ void kernel_normalized_list(float* output, int lim, int size)
{
     const int index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < size)
     {
        output[index] = (int)index - lim;
     }
}

void normalized_list(float* output, int lim, int size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_normalized_list<<<blocks, threads, 0, stream>>>(output, lim, size);   
}

__device__ float comp_hermite_iter(int n, float x)
{
    if (n == 0)
        return 1.0f;
    if (n == 1)
        return 2.0f * x;
    if (n > 1)
        return (2.0f * x * comp_hermite_iter(n - 1, x)) - (2.0f * (n - 1) * comp_hermite_iter(n - 2, x));
    return 0.0f;
}

__device__ float comp_gaussian(float x, float sigma)
{
    return 1 / (sigma * (sqrt(2 * M_PI))) * exp((-1 * x * x) / (2 * sigma * sigma));
}

__device__ float device_comp_dgaussian(float x, float sigma, int n)
{
    float A = pow((-1 / (sigma * sqrt((float)2))), n);
    float B = comp_hermite_iter(n, x / (sigma * sqrt((float)2)));
    float C = comp_gaussian(x, sigma);
    return A * B * C;
}

__global__ void kernel_comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] = device_comp_dgaussian(input[index], sigma, n);
    }
}


void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);
    kernel_comp_dgaussian<<<blocks, threads, 0, stream>>>(output, input, input_size, sigma, n);   
}


__global__ void convolutionKernel(const float* image, const float* kernel, float* output, 
                                  int width, int height, int kWidth, int kHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // Ensure we don't go out of image bounds

    float result = 0.0f;
    int kHalfWidth = kWidth / 2;
    int kHalfHeight = kHeight / 2;

    // Apply the convolution with replicate boundary behavior
    for (int ky = -kHalfHeight; ky <= kHalfHeight; ++ky) {
        for (int kx = -kHalfWidth; kx <= kHalfWidth; ++kx) {
            // Calculate the coordinates for the image
            int ix = x + kx;
            int iy = y + ky;

            // Replicate boundary behavior
            if (ix < 0) ix = 0;
            if (ix >= width) ix = width - 1;
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;

            float imageValue = image[iy * width + ix];
            float kernelValue = kernel[(ky + kHalfHeight) * kWidth + (kx + kHalfWidth)];
            result += imageValue * kernelValue;
        }
    }

    output[y * width + x] = result;
}

void applyConvolution(float* image, const float* kernel, 
                      int width, int height, int kWidth, int kHeight, cudaStream_t stream)
{
    float * d_output;
    cudaMalloc(&d_output, width * height * sizeof(float));


    // Définir la taille des blocs et de la grille
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    // Lancer le kernel
    convolutionKernel<<<gridSize, blockSize, 0, stream>>>(image, kernel, d_output, width, height, kWidth, kHeight);

    // Copier le résultat du GPU vers le CPU
    cudaMemcpy(image, d_output, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

    // Libérer la mémoire sur le GPU
    cudaFree(d_output);
}


void gaussian_imfilter_sep(float* input_output,
                            float* gpu_kernel_buffer,
                            int kernel_x_size,
                            int kernel_y_size,
                            const size_t frame_res,
                            float* convolution_buffer,
                            cuComplex* cuComplex_buffer,
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
    // This convolution method gives correct values compared to matlab
    applyConvolution(input_output,
                     gpu_kernel_buffer, 
                     std::sqrt(frame_res),
                     std::sqrt(frame_res),
                     kernel_x_size,
                     kernel_y_size,
                     stream);
}

__global__ void kernel_abs_lambda_division(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] = abs(lambda_1[index]) / abs(lambda_2[index]);
    }
}

void abs_lambda_division(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    
    kernel_abs_lambda_division<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res); 
}


__global__ void kernel_normalize(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] = sqrtf(powf(lambda_1[index], 2) + powf(lambda_2[index], 2));
    }
}

void normalize(float* output, float* lambda_1, float* lambda_2, uint frame_res, cudaStream_t stream) {
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_normalize<<<blocks, threads, 0, stream>>>(output, lambda_1, lambda_2, frame_res);
}

__global__ void kernel_If(float* output, size_t input_size, float* R_blob, float beta, float c, float *c_temp)
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

void If(float* output, size_t input_size, float* R_blob, float beta, float c, float *c_temp, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_If<<<blocks, threads, 0, stream>>>(output, input_size, R_blob, beta, c, c_temp);
}



__global__ void kernel_lambda_2_logical(float* output, size_t input_size, float* lambda_2)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] *= (lambda_2[index] <= 0.f ? 1 : 0);
    }
}

void lambda_2_logical(float* output, size_t input_size, float* lambda_2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);

    kernel_lambda_2_logical<<<blocks, threads, 0, stream>>>(output, input_size, lambda_2);
}

float* compute_I(float* input, float* g_mul, float A, uint frame_res, uint kernel_x_size, uint kernel_y_size, float* convolution_buffer, cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan, cudaStream_t stream) {
    float* I;
    cudaXMalloc(&I, frame_res * sizeof(float));
    cudaXMemcpyAsync(I, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    gaussian_imfilter_sep(I, g_mul, kernel_x_size, kernel_y_size, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);

    multiply_array_by_scalar(I, frame_res, A, stream);

    return I;
}

// Debuging: return float* to print the result to the screen
void vesselness_filter(float* output,
                        float* input, 
                        float sigma, 
                        float* g_xx_mul, 
                        float* g_xy_mul, 
                        float* g_yy_mul,
                        int kernel_x_size,
                        int kernel_y_size,
                        int frame_res, 
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cublasHandle_t cublas_handler,
                        cudaStream_t stream)
{
    int gamma = 1;
    float beta = 0.8f;

    float A = std::pow(sigma, gamma);

    float* Ixx = compute_I(input, g_xx_mul, A, frame_res, kernel_x_size, kernel_y_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);

    float* Ixy = compute_I(input, g_xy_mul, A, frame_res, kernel_x_size, kernel_y_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);

    float* Iyy = compute_I(input, g_yy_mul, A, frame_res, kernel_x_size, kernel_y_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);



    float* H;
    cudaMalloc(&H, frame_res * 3 * sizeof(float));


    int blockSize = 256;
    int numBlocks = (frame_res + blockSize - 1) / blockSize;
    prepareHessian<<<numBlocks, blockSize, 0, stream>>>(H, Ixx, Ixy, Iyy, frame_res);
    cudaXStreamSynchronize(stream);


    cudaXFree(Ixx);
    cudaXFree(Ixy);
    cudaXFree(Iyy);


    float* lambda_1 = new float[frame_res];
    cudaXMalloc(&lambda_1, frame_res * sizeof(float));
    cudaXMemset(lambda_1, 0, frame_res * sizeof(float));

    float* lambda_2 = new float[frame_res];
    cudaXMalloc(&lambda_2, frame_res * sizeof(float));
    cudaXMemset(lambda_2, 0, frame_res * sizeof(float));

    compute_eigen_values(H, frame_res, lambda_1, lambda_2, stream);


    cudaXFree(H);

    float* R_blob;
    cudaXMalloc(&R_blob, frame_res * sizeof(float));
    abs_lambda_division(R_blob, lambda_1, lambda_2, frame_res, stream);

    float *c_temp;
    cudaXMalloc(&c_temp, frame_res * sizeof(float));
    normalize(c_temp, lambda_1, lambda_2, frame_res, stream);

    int c_index;
    cublasStatus_t status = cublasIsamax(cublas_handler, frame_res, c_temp, 1, &c_index);

    float c;
    cudaMemcpy(&c, &c_temp[c_index - 1], sizeof(float), cudaMemcpyDeviceToHost);

    If(output, frame_res, R_blob, beta, c, c_temp, stream);

    cudaXFree(R_blob);
    cudaXFree(c_temp);
    cudaXFree(lambda_1);

    lambda_2_logical(output, frame_res, lambda_2, stream);

    cudaXFree(lambda_2);
}