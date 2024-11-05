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

// MIGHT NEED TO BE DELETED
// Useless for now !! Function to apply convolution with replicate padding
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

__global__ void kernel_normalize(float* output, float* lambda_1, float* lambda_2, size_t input_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] = sqrtf(powf(lambda_1[index], 2) + powf(lambda_2[index], 2));
    }
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

        //output[index] = expf(-(powf(R_blob[index], 2) / 2 * powf(beta, 2))) * (1 - expf(-(c_temp[index] / (2 * powf(*c, 2)))));
        //output[index] = exp(-(pow(R_blob[index], 2) / 2 * pow(beta, 2))) * (1 - exp(-(c_temp[index] / (2 * pow(1, 2)))));
    }
}

__global__ void kernel_lambda_2_logical(float* output, size_t input_size, float* lambda_2)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        output[index] *= (lambda_2[index] <= 0.f ? 1 : 0);
    }
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


    float* Ixx;
    cudaXMalloc(&Ixx, frame_res * sizeof(float));
    cudaXMemcpyAsync(Ixx, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);

    gaussian_imfilter_sep(Ixx, g_xx_mul, kernel_x_size, kernel_y_size, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXStreamSynchronize(stream);

    multiply_array_by_scalar(Ixx, frame_res, A, stream);
    cudaXStreamSynchronize(stream);


    float* Ixy;
    cudaXMalloc(&Ixy, frame_res * sizeof(float));
    cudaXMemcpyAsync(Ixy, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);

    gaussian_imfilter_sep(Ixy, g_xy_mul, kernel_x_size, kernel_y_size, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXStreamSynchronize(stream);

    multiply_array_by_scalar(Ixy, frame_res, A, stream);
    cudaXStreamSynchronize(stream);


    // Iyx is the same as Ixy, we can simply copy it
    float* Iyx;
    cudaXMalloc(&Iyx, frame_res * sizeof(float));
    cudaXMemcpyAsync(Iyx, Ixy, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);


    float* Iyy;
    cudaXMalloc(&Iyy, frame_res * sizeof(float));
    cudaXMemcpyAsync(Iyy, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);

    gaussian_imfilter_sep(Iyy, g_yy_mul, kernel_x_size, kernel_y_size, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXStreamSynchronize(stream);

    multiply_array_by_scalar(Iyy, frame_res, A, stream);
    cudaXStreamSynchronize(stream);


    float* H;
    cudaXMalloc(&H, frame_res * sizeof(float) * 4);

    cudaXMemcpyAsync(H, Ixx, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXMemcpyAsync(H + frame_res, Ixy, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXMemcpyAsync(H + frame_res * 2, Iyx, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXMemcpyAsync(H + frame_res * 3, Iyy, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    cudaXStreamSynchronize(stream);


    print_in_file(Ixx, frame_res, "ixx", stream);
    print_in_file(Ixy, frame_res, "ixy", stream);
    print_in_file(Iyx, frame_res, "iyx", stream);
    print_in_file(Iyy, frame_res, "iyy", stream);

    cudaXFree(Ixx);
    cudaXFree(Ixy);
    cudaXFree(Iyx);
    cudaXFree(Iyy);


    float* lambda_1;
    cudaXMalloc(&lambda_1, frame_res * sizeof(float));
    cudaXMemset(lambda_1, 0, frame_res * sizeof(float));
    float* lambda_2;
    cudaXMalloc(&lambda_2, frame_res * sizeof(float));
    cudaXMemset(lambda_2, 0, frame_res * sizeof(float));

    cudaXStreamSynchronize(stream);


    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    dim3 blockSize(16, 16);
    dim3 gridSize((std::sqrt(frame_res) + blockSize.x - 1) / blockSize.x, (std::sqrt(frame_res) + blockSize.y - 1) / blockSize.y);

    kernel_4D_eigenvalues<<<gridSize, blockSize, 0, stream>>>(H, lambda_1, lambda_2, std::sqrt(frame_res), std::sqrt(frame_res));
    
    cudaXStreamSynchronize(stream);

    print_in_file(lambda_1, frame_res, "lambda_1", stream);
    print_in_file(lambda_2, frame_res, "lambda_2", stream);

    // cudaXFree(H);

    // float* R_blob;
    // cudaXMalloc(&R_blob, frame_res * sizeof(float));
    // threads = get_max_threads_1d();
    // blocks = map_blocks_to_problem(frame_res, threads);
    // kernel_abs_lambda_division<<<blocks, threads, 0, stream>>>(R_blob, lambda_1, lambda_2, frame_res);
    // cudaXStreamSynchronize(stream);


    // float *c_temp;
    // cudaXMalloc(&c_temp, frame_res * sizeof(float));
    // threads = get_max_threads_1d();
    // blocks = map_blocks_to_problem(frame_res, threads);
    // kernel_normalize<<<blocks, threads, 0, stream>>>(c_temp, lambda_1, lambda_2, frame_res);
    // cudaXStreamSynchronize(stream);


    // int c_index;
    // cublasStatus_t status = cublasIsamax(cublas_handler, frame_res, c_temp, 1, &c_index);
    // cudaXStreamSynchronize(stream);
    // float c;
    // cudaMemcpy(&c, &c_temp[c_index - 1], sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "c : " << c << std::endl;

    // threads = get_max_threads_1d();
    // blocks = map_blocks_to_problem(frame_res, threads);
    // kernel_If<<<blocks, threads, 0, stream>>>(output, frame_res, R_blob, beta, c, c_temp);
    // cudaXStreamSynchronize(stream);

    // test_filter = new float[frame_res];
    // cudaXMemcpyAsync(test_filter,
    //     output,
    //     frame_res * sizeof(float),
    //     cudaMemcpyDeviceToHost,
    //     stream);
    // write1DFloatArrayToFile(test_filter,
    //     sqrt(frame_res),
    //     sqrt(frame_res),
    //     "test_filter_output.txt");

    // cudaXFree(R_blob);
    // cudaXFree(c_temp);
    // cudaXFree(lambda_1);

    // threads = get_max_threads_1d();
    // blocks = map_blocks_to_problem(frame_res, threads);
    // kernel_lambda_2_logical<<<blocks, threads, 0, stream>>>(output, frame_res, lambda_2);
    // cudaXStreamSynchronize(stream);

    // cudaXFree(lambda_2);
}