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


static void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
            {
                outFile << " "; // Separate values in a row by a space
            }
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

void gaussian_imfilter_sep(float* input_img,
                            float* kernel,
                            const size_t frame_res,
                            cuComplex* gpu_kernel_buffer,
                            float* convolution_buffer, 
                            cuComplex* cuComplex_buffer, 
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
    float *cpu_kernel = new float[frame_res];
    cudaXMemcpy(cpu_kernel, kernel, frame_res * sizeof(float), cudaMemcpyDeviceToHost);
    write1DFloatArrayToFile(cpu_kernel, std::sqrt(frame_res), std::sqrt(frame_res), "kernel_2.txt");
    cudaXMemsetAsync(gpu_kernel_buffer, 0, frame_res * sizeof(cuComplex), stream);
    cudaSafeCall(cudaMemcpy2DAsync(gpu_kernel_buffer,
                                   sizeof(cuComplex),
                                   kernel,
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyDeviceToDevice,
                                   stream));

    convolution_kernel(input_img,
                        convolution_buffer,
                        cuComplex_buffer,
                        convolution_plan,
                        frame_res,
                        gpu_kernel_buffer,
                        false,
                        stream);
    cudaXStreamSynchronize(stream);
}

void multiply_by_float(float* vect, float num, int frame_size)
{
    for (int i = 0; i < frame_size; i++)
    {
        vect[i] *= num;
    }
}

void vesselness_filter(float* output,
                        float* input, 
                        float sigma, 
                        float* g_xx_mul, 
                        float* g_xy_mul, 
                        float* g_yy_mul,
                        int frame_res, 
                        cuComplex* gpu_kernel_buffer,
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cudaStream_t stream)
{
    int gamma = 1;

    float A = std::pow(sigma, gamma);

    float* input_copy;
    cudaXMalloc(&input_copy, frame_res * sizeof(float));
    cudaXMemcpy(input_copy, input, frame_res * sizeof(float));

    gaussian_imfilter_sep(input_copy, g_xx_mul, frame_res, gpu_kernel_buffer, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXMemcpy(output, input_copy, frame_res * sizeof(float));
    //multiply_by_float(output, A, frame_res);

    // float* Ixy = gaussian_imfilter_sep(input, g_xy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    // multiply_by_float(Ixy, A, frame_res);

    // float* Iyx = new float[frame_res];
    // for (size_t i = 0; i < frame_res; ++i) {
    //     Iyx[i] = Ixy[i];
    // }

    // float* Iyy = gaussian_imfilter_sep(input, g_yy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    // multiply_by_float(Iyy, A, frame_res);
}