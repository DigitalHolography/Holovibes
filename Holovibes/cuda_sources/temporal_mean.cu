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

__global__ void kernel_substract_first_image(float* sum_image, float* all_images, const int current_image, const int time_window, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        sum_image[index] -= all_images[frame_size * ((current_image - 1) % time_window) + index];
}

__global__ void kernel_add_img_to_sum(float* sum_image, float* current_image, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        sum_image[index] += current_image[index];
}

__global__ void kernel_compute_mean(float* sum_image, float* act_image, const int time_window, const uint frame_size)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
    {
        act_image[index] = sum_image[index] / time_window;
    }
}

void temporal_mean(float* input_output,
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
        cudaXMemcpyAsync(image_buffer, input_output, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaXStreamSynchronize(stream);
        cudaXMemcpyAsync(image_sum, input_output, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
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
        kernel_add_img_to_sum<<<blocks, threads, 0, stream>>>(image_sum, input_output, frame_size);
        cudaXMemcpyAsync(image_buffer + frame_size * ((*current_image - 1) % time_window), input_output, frame_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaXStreamSynchronize(stream);
        
        if (*current_image >= time_window)
        {
            threads = get_max_threads_1d();
            blocks = map_blocks_to_problem(frame_size, threads);
            kernel_compute_mean<<<blocks, threads, 0, stream>>>(image_sum, input_output, time_window, frame_size);
            cudaXStreamSynchronize(stream);
        }
    }

}