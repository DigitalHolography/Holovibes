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

void gaussian_imfilter_sep(float* input_output,
                            cuComplex* gpu_kernel_buffer,
                            const size_t frame_res,
                            float* convolution_buffer, 
                            cuComplex* cuComplex_buffer, 
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
    convolution_kernel(input_output,
                        convolution_buffer,
                        cuComplex_buffer,
                        convolution_plan,
                        frame_res,
                        gpu_kernel_buffer,
                        false,
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


// Kernel pour calculer le min et le max d'un tableau
__global__ void findMinMax(const float *input, float *min, float *max, int n) {
    extern __shared__ float shared_data[];
    float *s_min = shared_data;
    float *s_max = shared_data + blockDim.x;
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialiser les valeurs partagées
    if (index < n) {
        s_min[tid] = input[index];
        s_max[tid] = input[index];
    } else {
        s_min[tid] = FLT_MAX;
        s_max[tid] = -FLT_MAX;
    }
    __syncthreads();

    // Réduction pour trouver le min et le max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && index + s < n) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        }
        __syncthreads();
    }

    // Stocker le min et le max globalement
    if (tid == 0) {
        atomicMin((int *)min, __float_as_int(s_min[0]));
        atomicMax((int *)max, __float_as_int(s_max[0]));
    }
}

// Kernel pour normaliser chaque élément entre 0 et 255
__global__ void normalizeTo0255(float *data, float min, float max, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n && max > min) {
        data[index] = ((data[index] - min) / (max - min)) * 255.0f;
    }
}

// Fonction principale de normalisation entre 0 et 255
void normalizeArrayTo0255(float *d_data, int n) {
    // Allocation pour stocker le min et le max sur le device
    float *d_min, *d_max;
    cudaMalloc((void **)&d_min, sizeof(float));
    cudaMalloc((void **)&d_max, sizeof(float));

    // Initialiser min et max
    float h_min = FLT_MAX, h_max = -FLT_MAX;
    cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Trouver le min et le max
    findMinMax<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(d_data, d_min, d_max, n);

    // Copier le min et le max du device vers le host
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    // Appliquer la normalisation entre 0 et 255
    normalizeTo0255<<<gridSize, blockSize>>>(d_data, h_min, h_max, n);

    // Libérer la mémoire
    cudaFree(d_min);
    cudaFree(d_max);
}

void vesselness_filter(float* output,
                        float* input, 
                        float sigma, 
                        cuComplex* g_xx_mul, 
                        cuComplex* g_xy_mul, 
                        cuComplex* g_yy_mul,
                        int frame_res, 
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cublasHandle_t cublas_handler,
                        cudaStream_t stream)
{
    normalizeArrayTo0255(input, frame_res);

    int gamma = 1;
    float beta = 0.8f;

    float A = std::pow(sigma, gamma);

    float* Ixx;
    cudaXMalloc(&Ixx, frame_res * sizeof(float));
    cudaXMemcpyAsync(Ixx, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);
    gaussian_imfilter_sep(Ixx, g_xx_mul, frame_res,convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXStreamSynchronize(stream);
    multiply_array_by_scalar(Ixx, frame_res, A, stream);
    cudaXStreamSynchronize(stream);



    float* Ixy;
    cudaXMalloc(&Ixy, frame_res * sizeof(float));
    cudaXMemcpyAsync(Ixy, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);
    gaussian_imfilter_sep(Ixy, g_xy_mul, frame_res,convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    cudaXStreamSynchronize(stream);
    multiply_array_by_scalar(Ixy, frame_res, A, stream);
    cudaXStreamSynchronize(stream);


    float* Iyx;
    cudaXMalloc(&Iyx, frame_res * sizeof(float));
    cudaXMemcpyAsync(Iyx, Ixy, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);

    float* Iyy;
    cudaXMalloc(&Iyy, frame_res * sizeof(float));
    cudaXMemcpyAsync(Iyy, input, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaXStreamSynchronize(stream);
    gaussian_imfilter_sep(Iyy, g_yy_mul, frame_res,convolution_buffer, cuComplex_buffer, convolution_plan, stream);
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

    cudaXFree(Ixx);
    cudaXFree(Ixy);
    cudaXFree(Iyx);
    cudaXFree(Iyy);

    // H works here 

    float* lambda_1;
    cudaXMalloc(&lambda_1, frame_res * sizeof(float));
    cudaXMemset(lambda_1, 0, frame_res * sizeof(float));
    float* lambda_2;
    cudaXMalloc(&lambda_2, frame_res * sizeof(float));
    cudaXMemset(lambda_2, 0, frame_res * sizeof(float));

    cudaXStreamSynchronize(stream);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
     // Define grid and block dimensions
    dim3 blockSize(16, 16); // Block size (16x16)
    dim3 gridSize((std::sqrt(frame_res) + blockSize.x - 1) / blockSize.x, (std::sqrt(frame_res) + blockSize.y - 1) / blockSize.y);
    kernel_4D_eigenvalues<<<gridSize, blockSize, 0, stream>>>(H, lambda_1, lambda_2, std::sqrt(frame_res), std::sqrt(frame_res));
    cudaXStreamSynchronize(stream);

    float *test_filter = new float[frame_res];
    cudaXMemcpyAsync(test_filter,
        lambda_2,
        frame_res * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    write1DFloatArrayToFile(test_filter,
        sqrt(frame_res),
        sqrt(frame_res),
        "test_filter_lambda_2.txt");

    cudaXFree(H);

    float* R_blob;
    cudaXMalloc(&R_blob, frame_res * sizeof(float));
    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_abs_lambda_division<<<blocks, threads, 0, stream>>>(R_blob, lambda_1, lambda_2, frame_res);
    cudaXStreamSynchronize(stream);


    float *c_temp;
    cudaXMalloc(&c_temp, frame_res * sizeof(float));
    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_normalize<<<blocks, threads, 0, stream>>>(c_temp, lambda_1, lambda_2, frame_res);
    cudaXStreamSynchronize(stream);

    test_filter = new float[frame_res];
    cudaXMemcpyAsync(test_filter,
        c_temp,
        frame_res * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    write1DFloatArrayToFile(test_filter,
        sqrt(frame_res),
        sqrt(frame_res),
        "test_filter_ctemp.txt");

    int c_index;
    cublasStatus_t status = cublasIsamax(cublas_handler, frame_res, c_temp, 1, &c_index);
    cudaXStreamSynchronize(stream);
    float c;
    cudaMemcpy(&c, &c_temp[c_index - 1], sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "c : " << c << std::endl;

    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_If<<<blocks, threads, 0, stream>>>(output, frame_res, R_blob, beta, c, c_temp);
    cudaXStreamSynchronize(stream);

    test_filter = new float[frame_res];
    cudaXMemcpyAsync(test_filter,
        output,
        frame_res * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    write1DFloatArrayToFile(test_filter,
        sqrt(frame_res),
        sqrt(frame_res),
        "test_filter_output.txt");

    cudaXFree(R_blob);
    cudaXFree(c_temp);
    cudaXFree(lambda_1);

    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_lambda_2_logical<<<blocks, threads, 0, stream>>>(output, frame_res, lambda_2);
    cudaXStreamSynchronize(stream);

    cudaXFree(lambda_2);
}