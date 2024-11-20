#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>

#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "vascular_pulse.cuh"

__global__ void kernel_divide_constant(float* vascular_pulse, int value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] /= value;
}

void divide_constant(float* vascular_pulse, int value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
}

__global__ void kernel_divide(float* vascular_pulse, float* value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] /= value[index];
}

void divide(float* vascular_pulse, float* value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
}

__global__ void kernel_multiply_constant(float* vascular_pulse, float* value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] *= (*value);
}

void multiply_constant(float* vascular_pulse, float* value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
}

float compute_mean(float* vascular_pulse, size_t size)
{
    thrust::device_ptr<float> d_array(vascular_pulse);

    float sum = thrust::reduce(d_array, d_array + size, 0.0f, thrust::plus<float>());

    return sum / size;
}

__global__ void kernel_subtract_constant(float* output, float* input, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input[index] - value;
}

void subtract_constant(float* output, float* input, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_constant<<<blocks, threads, 0, stream>>>(output, input, value, size);
}

__global__ void kernel_multiply_three_vectors(float* output, float* input1, float* input2, float* input3, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input1[index] * input2[index] * input3[index];
}

void multiply_three_vectors(
    float* output, float* input1, float* input2, float* input3, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_three_vectors<<<blocks, threads, 0, stream>>>(output, input1, input2, input3, size);
}

__global__ void kernel_computeMean(
    const float* M0, const float* vascularPulse, float* result, 
    int rows, int cols, int depth
) {
    // Calcul des indices globaux
    int x = blockIdx.x * blockDim.x + threadIdx.x; // index de ligne
    int y = blockIdx.y * blockDim.y + threadIdx.y; // index de colonne

    if (x < rows && y < cols) {
        float sum = 0.0f;

        // Somme sur la 3ème dimension
        for (int z = 0; z < depth; ++z) {
            int index3D = x * cols + y + z * rows * cols;
            sum += M0[index3D] * vascularPulse[z];
        }

        // Stocker la moyenne dans le tableau résultat
        result[x * cols + y] = sum / depth;
    }
}

void computeMean(const float* M0, const float* vascularPulse, float* result, 
    int rows, int cols, int depth, cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, 
                  (cols + blockSize.y - 1) / blockSize.y);

    // Lancer le kernel
    kernel_computeMean<<<gridSize, blockSize, 0, stream>>>(M0, vascularPulse, result, rows, cols, depth);
}

__global__ void kernel_compute_std(const float* input, float* output, int size, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float mean = 0.0f;
        float variance = 0.0f;

        // Compute mean along the third dimension
        for (int k = 0; k < depth; ++k) {
            mean += input[idx + depth * k];
        }
        mean /= depth;

        // Compute variance along the third dimension
        for (int k = 0; k < depth; ++k) {
            float diff = input[idx + depth * k] - mean;
            variance += diff * diff;
        }
        variance /= depth;

        // Store the standard deviation in the output array
        output[idx] = sqrt(variance);
    }
}

void compute_std(const float* input, float* output, int size, int depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_compute_std<<<blocks, threads, 0, stream>>>(input, output, size, depth);
}



void compute_first_correlation(float* output,
                               float* M0_ff_video_centered,
                               float* vascular_pulse,
                               int nnz_mask_vesslness_clean,
                               size_t length_video,
                               size_t image_size,
                               cudaStream_t stream) // Size here is future time window
{

    float* vascular_pulse_copy;
    cudaXMalloc(&vascular_pulse_copy, sizeof(float) * length_video);
    cudaXMemcpyAsync(vascular_pulse_copy,
                     vascular_pulse,
                     sizeof(float) * length_video,
                     cudaMemcpyDeviceToDevice,
                     stream);

    divide_constant(vascular_pulse_copy, nnz_mask_vesslness_clean, length_video, stream);


    float* vascular_pulse_centered;
    cudaXMalloc(&vascular_pulse_centered,
                506 * sizeof(float)); // need to be replaced with time window (it's because csv)

    float vascular_mean = compute_mean(vascular_pulse_copy, length_video);
    subtract_constant(vascular_pulse_centered, vascular_pulse_copy, vascular_mean, length_video, stream);

    // TODO: la suite (le calcul de R_vascularPulse)
    computeMean(M0_ff_video_centered, vascular_pulse_centered, output, 512, 512, 506, stream);
    
    float *std_M0_ff_video_centered;
    cudaXMalloc(&std_M0_ff_video_centered, sizeof(float) * 512 * 512);
    compute_std(M0_ff_video_centered, std_M0_ff_video_centered, 512 * 512, 506, stream);

    float *std_vascular_pulse_centered;
    cudaXMalloc(&std_vascular_pulse_centered, sizeof(float));
    compute_std(vascular_pulse_centered, std_vascular_pulse_centered, 1, 506, stream);

    multiply_constant(std_M0_ff_video_centered, std_vascular_pulse_centered, 512 * 512, stream);

    divide(output, std_M0_ff_video_centered, 512 * 512, stream);
    
    cudaXFree(std_M0_ff_video_centered);
    cudaXFree(std_vascular_pulse_centered);
    cudaXFree(vascular_pulse_centered);
    cudaXFree(vascular_pulse_copy);
}