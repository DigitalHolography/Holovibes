#include <stdio.h>
#include <iostream>
#include <fstream>

#include "tools_hsv.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "percentile.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"
#include "reduce.cuh"
#include "unique_ptr.hh"
#include "logger.hh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

__global__ void kernel_add_frame_to_sum(const float* const new_frame, const size_t frame_size, float* const sum_image)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frame_size)
        sum_image[idx] += new_frame[idx];
}

void add_frame_to_sum(const float* const new_frame, const size_t size, float* const sum_image, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_add_frame_to_sum<<<blocks, threads, 0, stream>>>(new_frame, size, sum_image);
}

__global__ void kernel_subtract_frame_from_sum(const float* old_frame, const size_t frame_size, float* const sum_image)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frame_size)
        sum_image[idx] -= old_frame[idx];
}

void subtract_frame_from_sum(const float* const new_frame,
                             const size_t size,
                             float* const sum_image,
                             cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_frame_from_sum<<<blocks, threads, 0, stream>>>(new_frame, size, sum_image);
}

__global__ void kernel_compute_mean(float* output, float* input, const size_t time_window, const size_t frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] = input[index] / time_window;
}

void compute_mean(float* output, float* input, const size_t time_window, const size_t frame_size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size, threads);
    kernel_compute_mean<<<blocks, threads, 0, stream>>>(output, input, time_window, frame_size);
}

__global__ void kernel_compute_mean_1_2(float* const output, const float* const input, const int frame_size, const int frame_nb_)
{
    extern __shared__ float shared_data[];

    // Each block processes one image, each thread processes one element
    int image_index = blockIdx.x;
    if (image_index >= frame_nb_) return;  

    int tid = threadIdx.x;

    // Initialize partial sum for each thread in shared memory
    shared_data[tid] = 0.0f;

    // Accumulate sum within each thread
    for (int i = tid; i < frame_size; i += blockDim.x)
    {
        shared_data[tid] += input[image_index * frame_size + i];
    }

    // Synchronize threads to ensure all have written their partial sum
    __syncthreads();

    // Perform parallel reduction within the block to get the total sum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
            shared_data[tid] += shared_data[tid + stride];
        __syncthreads();
    }

    // Write the mean to the output buffer by the first thread of each block
    if (tid == 0)
        output[image_index] = shared_data[0] / frame_size;
}

void compute_mean_1_2(float* const output, const float* const input, const size_t frame_size, const size_t frame_nb, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_nb, threads);
    size_t sharedMemSize = threads / blocks * sizeof(float);
    kernel_compute_mean_1_2<<<blocks, threads, sharedMemSize, stream>>>(output, input, frame_size, frame_nb);
}

__global__ void
kernel_image_centering(float* output, const float* m0_video_frame, const float* m0_img, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] = m0_video_frame[index] - m0_img[index];
}

void image_centering(
    float* output, const float* m0_img, const float* m0_video_frame, const uint frame_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size, threads);
    kernel_image_centering<<<blocks, threads, 0, stream>>>(output, m0_video_frame, m0_img, frame_size);
    cudaXStreamSynchronize(stream);
}