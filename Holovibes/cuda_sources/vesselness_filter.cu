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

// Function to apply convolution with replicate padding
void applyConvolutionWithReplicatePadding(const float* image, float* output, int imgWidth, int imgHeight,
                                          const float* kernel, int kernelWidth, int kernelHeight,
                                          bool divideConvolution = false) {
    int padWidth = kernelWidth / 2;
    int padHeight = kernelHeight / 2;

    // Calculate the sum of the kernel for normalization
    float kernelSum = 0.0f;
    for (int i = 0; i < kernelHeight; ++i) {
        for (int j = 0; j < kernelWidth; ++j) {
            kernelSum += kernel[i * kernelWidth + j];
        }
    }

    // Iterate over each pixel in the output image
    for (int y = 0; y < imgHeight; ++y) {
        for (int x = 0; x < imgWidth; ++x) {
            float sum = 0.0f;

            // Convolution operation with replicate padding
            for (int ky = -padHeight; ky <= padHeight; ++ky) {
                for (int kx = -padWidth; kx <= padWidth; ++kx) {
                    int imgX = std::min(std::max(x + kx, 0), imgWidth - 1);
                    int imgY = std::min(std::max(y + ky, 0), imgHeight - 1);
                    int kernelIndex = (ky + padHeight) * kernelWidth + (kx + padWidth);
                    int imageIndex = imgY * imgWidth + imgX;

                    // Apply the kernel
                    sum += image[imageIndex] * kernel[kernelIndex];
                }
            }

            // If division is enabled, normalize the convolved value
            if (divideConvolution && kernelSum != 0.0f) {
                output[y * imgWidth + x] = sum / kernelSum;
            } else {
                output[y * imgWidth + x] = sum;
            }
        }
    }
}

__global__ void kernel_normalized_list(float* output, int lim, int size)
{
     const uint index = blockIdx.x * blockDim.x + threadIdx.x;
     if (index < size)
     {
        output[index] = index - lim;
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
    if (n < 0)
        return 0.0f;
    if (n == 0)
        return 1.0f;
    if (n == 1)
        return 2.0f * x;

    float h_n_minus_2 = 1.0f;           // H_0(x)
    float h_n_minus_1 = 2.0f * x;       // H_1(x)
    float h_n = 0.0f;

    for (int i = 2; i <= n; ++i)
    {
        h_n = (2.0f * x * h_n_minus_1) - (2.0f * (i - 1) * h_n_minus_2);
        h_n_minus_2 = h_n_minus_1;
        h_n_minus_1 = h_n;
    }

    return h_n;
}

__device__ float comp_gaussian(float x, float sigma)
{
    return 1 / (sigma * (sqrtf(2 * M_PI))) * expf((-1 * x * x) / (2 * sigma * sigma));
}

__device__ float device_comp_dgaussian(float x, float sigma, int n)
{
    float A = powf((-1 / (sigma * sqrtf(2))), n);
    float B = comp_hermite_iter(n, x / (sigma * sqrtf(2)));
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

float* gaussian_imfilter_sep(float* input_img, 
                            float* kernel,
                            size_t kernel_height,
                            size_t kernel_width,
                            const size_t frame_res, 
                            float* convolution_buffer, 
                            cuComplex* cuComplex_buffer, 
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
   
    // float* gpu_output;
    // cudaXMalloc(&gpu_output, frame_res * sizeof(float));
    // cudaXMemcpyAsync(gpu_output, input_img, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // cudaXStreamSynchronize(stream);

    // cuComplex* output_complex;
    // cudaXMalloc(&output_complex, frame_res * sizeof(cuComplex));
    // cudaXMemset(output_complex, 0, frame_res * sizeof(cuComplex));
    // load_kernel_in_GPU(output_complex, kernel, frame_res, stream);
    // cudaXStreamSynchronize(stream);

    // convolution_kernel(gpu_output,
    //                     convolution_buffer,
    //                     cuComplex_buffer,
    //                     convolution_plan,
    //                     frame_res,
    //                     output_complex,
    //                     false,
    //                     stream);
    // cudaXStreamSynchronize(stream);

    float *input_copy = new float[frame_res];
    cudaXMemcpyAsync(input_copy, input_img, frame_res * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);

    float *output = new float[frame_res];
    applyConvolutionWithReplicatePadding(input_copy, output, std::sqrt(frame_res), std::sqrt(frame_res), kernel, kernel_width, kernel_height, true);

    free(input_copy);

    // float *res = new float[frame_res];
    // cudaXMemcpyAsync(res, gpu_output, frame_res * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // cudaXStreamSynchronize(stream);
    
    return output;
}

void multiply_by_float(float* vect, float num, int frame_size)
{
    for (int i = 0; i < frame_size; i++)
    {
        vect[i] *= num;
    }
}


float* vesselness_filter(float* input, 
                        float sigma, 
                        float* g_xx_mul, 
                        float* g_xy_mul, 
                        float* g_yy_mul,
                        size_t kernel_height,
                        size_t kernel_width,
                        int frame_res, 
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cudaStream_t stream)
{
    int gamma = 1;

    float A = std::pow(sigma, gamma);

    float* Ixx = gaussian_imfilter_sep(input, g_xx_mul, kernel_height, kernel_width, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Ixx, A, frame_res);

    // float* Ixy = gaussian_imfilter_sep(input, g_xy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    // multiply_by_float(Ixy, A, frame_res);

    // float* Iyx = new float[frame_res];
    // for (size_t i = 0; i < frame_res; ++i) {
    //     Iyx[i] = Ixy[i];
    // }

    // float* Iyy = gaussian_imfilter_sep(input, g_yy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    // multiply_by_float(Iyy, A, frame_res);

    return Ixx;
}