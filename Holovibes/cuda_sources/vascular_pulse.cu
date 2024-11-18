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
    {
        vascular_pulse[index] /= value;
    }
}

void divide_constant(float* vascular_pulse, int value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
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
    {
        output[index] = input[index] - value;
    }
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
    {
        output[index] = input1[index] * input2[index] * input3[index];
    }
}

void multiply_three_vectors(
    float* output, float* input1, float* input2, float* input3, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_three_vectors<<<blocks, threads, 0, stream>>>(output, input1, input2, input3, size);
}

void compute_first_correlation(float* output,
                               float* image_centered,
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

    float* R_VascularPulse;
    cudaXMalloc(&R_VascularPulse, image_size * sizeof(float));

    // TODO: la suite (le calcul de R_vascularPulse)

    cudaXFree(vascular_pulse_centered);
    cudaXFree(vascular_pulse_copy);
}