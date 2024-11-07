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



void compute_barycentre(float* output,
                        float* temporal_mean_img,
                        float* temporal_mean_video,
                        size_t size,
                        cudaStream_t stream)
{
    compute_multiplication(output, temporal_mean_img, temporal_mean_video, size, stream);

    int index_max = find_max_thrust(output, size);
    int index_min = find_min_thrust(output, size);

    float* cercleMask;
    cudaXMalloc(&cercleMask, sizeof(float) * size);
    compute_circle_mask(cercleMask,
        index_max % (int) std::floor(std::sqrt(size)),
        std::floor(index_max / std::sqrt(size)),
        0.1 * (std::sqrt(size) + std::sqrt(size)) / 2,
        std::sqrt(size),
        std::sqrt(size),
        stream
    );
    float* cercleMask2;
    cudaXMalloc(&cercleMask2, sizeof(float) * size);
    compute_circle_mask(cercleMask2,
        index_min % (int) std::floor(std::sqrt(size)),
        std::floor(index_min / std::sqrt(size)),
        0.1 * (std::sqrt(size) + std::sqrt(size)) / 2,
        std::sqrt(size),
        std::sqrt(size),
        stream
    );
    apply_mask_or(cercleMask, cercleMask2, std::sqrt(size), std::sqrt(size), stream);
    cudaXFree(cercleMask2);
    cudaXFree(cercleMask);
    //cudaXMemcpy(output, cercleMask, size * sizeof(float), cudaMemcpyDeviceToDevice);

}
