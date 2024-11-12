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

__global__ void kernel_substract_constant(float* output, float* input, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index] = input[index] - value;
    }
}

void substract_constant(float* output, float* input, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_substract_constant<<<blocks, threads, 0, stream>>>(output, input, value, size);
}


void compute_first_correlation(float* output, float* vascular_pulse, int nnz_mask_vesslness_clean, size_t size, cudaStream_t stream)    // Size here is future time window
{
    float* vascular_pulse_copy;
    cudaXMalloc(&vascular_pulse_copy, sizeof(float) * size);
    cudaXMemcpyAsync(vascular_pulse_copy, vascular_pulse, sizeof(float) * size, cudaMemcpyDeviceToDevice, stream);
    
    divide_constant(vascular_pulse_copy, nnz_mask_vesslness_clean, size, stream);

    float* vascular_pulse_centered;
    cudaXMalloc(&vascular_pulse_centered, 506 * sizeof(float)); // need to be replaced with time window (it's because csv)

    float vascular_mean = compute_mean(vascular_pulse_copy, size);
    substract_constant(vascular_pulse_centered, vascular_pulse_copy, vascular_mean, size, stream);

    cudaXFree(vascular_pulse_centered);
    cudaXFree(vascular_pulse_copy);
}