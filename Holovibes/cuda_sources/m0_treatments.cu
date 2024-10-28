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

__global__ void kernel_substract_first_image(float* output, float* input, const int current_image, const int time_window, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] -= input[frame_size * ((current_image - 1) % time_window) + index];
}

__global__ void kernel_add_img_to_sum(float* output, float* input, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] += input[index];
}

__global__ void kernel_compute_mean(float* output, float* input, const int time_window, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] = input[index] / time_window;
}

void temporal_mean(float* output,
                    float* input,
                        int* current_image,
                        float* image_buffer,
                        float* image_sum,
                        const int time_window,
                        const uint frame_size,
                        const cudaStream_t stream)
{
    if (*current_image >= 2 * time_window)
        *current_image -= time_window;
    (*current_image)++;
    if (*current_image == 1)
    {
        cudaXMemcpyAsync(image_buffer, input, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaXStreamSynchronize(stream);
        cudaXMemcpyAsync(image_sum, input, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaXStreamSynchronize(stream);
    }
    else
    {
        if (*current_image >= time_window)
        {
            uint threads = get_max_threads_1d();
            uint blocks = map_blocks_to_problem(frame_size, threads);
            kernel_substract_first_image<<<blocks, threads, 0, stream>>>(image_sum, image_buffer, *current_image, time_window, frame_size);
            cudaXStreamSynchronize(stream);
        }

        uint threads = get_max_threads_1d();
        uint blocks = map_blocks_to_problem(frame_size, threads);
        kernel_add_img_to_sum<<<blocks, threads, 0, stream>>>(image_sum, input, frame_size);
        cudaXMemcpyAsync(image_buffer + frame_size * ((*current_image - 1) % time_window), input, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaXStreamSynchronize(stream);
        
        if (*current_image >= time_window)
        {
            threads = get_max_threads_1d();
            blocks = map_blocks_to_problem(frame_size, threads);
            kernel_compute_mean<<<blocks, threads, 0, stream>>>(output, image_sum, time_window, frame_size);
            cudaXStreamSynchronize(stream);
        }
    }

}

__global__ void kernel_centering(float* output, const float* m0_video, const float* m0_img, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        output[index] = m0_video[index] - m0_img[index];
}

void image_centering(float* output, const float* m0_video, const float* m0_img, const uint frame_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size, threads);
    kernel_centering<<<blocks, threads, 0, stream>>>(output, m0_video, m0_img, frame_size);
    cudaXStreamSynchronize(stream);
}