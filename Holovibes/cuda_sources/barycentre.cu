#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "barycentre.cuh"
#include <cmath>

#define CIRCLE_MASK_RADIUS 0.07

__global__ void kernel_compute_multiplication(float* output, float* A, float* B, size_t size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        output[index] = A[index] * B[index];
    }
}

void compute_multiplication(float* output, float* A, float* B, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_compute_multiplication<<<blocks, threads, 0, stream>>>(output, A, B, size);
}

int find_max_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
    return max_ptr - dev_ptr;
}

int find_min_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + size);
    return min_ptr - dev_ptr;
}
 

int compute_barycentre_circle_mask(float* output,
                                    float* input,
                                    size_t size,
                                    cudaStream_t stream, 
                                    int CRV_index)
{
    int barycentre_index = find_max_thrust(input, size);
    if (CRV_index == -1)
        CRV_index = find_min_thrust(input, size); 

    compute_circle_mask(output,
        barycentre_index % (int) std::floor(std::sqrt(size)),
        std::floor(barycentre_index / std::sqrt(size)),
        CIRCLE_MASK_RADIUS * (std::sqrt(size) + std::sqrt(size)) / 2,
        std::sqrt(size),
        std::sqrt(size),
        stream
    );

    // circle_mask_min is CRV
    float* circle_mask_min;
    cudaXMalloc(&circle_mask_min, sizeof(float) * size);
    compute_circle_mask(circle_mask_min,
        CRV_index % (int) std::floor(std::sqrt(size)),
        std::floor(CRV_index / std::sqrt(size)),
        CIRCLE_MASK_RADIUS * (std::sqrt(size) + std::sqrt(size)) / 2,
        std::sqrt(size),
        std::sqrt(size),
        stream
    );

    apply_mask_or(output, circle_mask_min, std::sqrt(size), std::sqrt(size), stream);
    cudaXStreamSynchronize(stream);
    cudaXFree(circle_mask_min);

    return CRV_index;
}
