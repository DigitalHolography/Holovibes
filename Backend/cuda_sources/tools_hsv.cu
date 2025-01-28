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

__global__ void kernel_threshold_top_bottom(float* output, const float tmin, const float tmax, const uint frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        output[id] = fminf(output[id], tmax);
        output[id] = fmaxf(output[id], tmin);
    }
}

void threshold_top_bottom(
    float* output, const float tmin, const float tmax, const uint frame_res, const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_threshold_top_bottom<<<blocks, threads, 0, stream>>>(output, tmin, tmax, frame_res);
    cudaCheckError();
}

void apply_percentile_and_threshold(float* gpu_arr,
                                    uint frame_res,
                                    uint width,
                                    uint height,
                                    float low_threshold,
                                    float high_threshold,
                                    const cudaStream_t stream)
{
    float percent_out[2];
    const float percent_in_h[2] = {low_threshold, high_threshold};
    auto exec_policy = thrust::cuda::par.on(stream);

    compute_percentile_xy_view(gpu_arr, width, height, 0, percent_in_h, percent_out, 2, 0.0f, false, stream);

    threshold_top_bottom(gpu_arr, percent_out[0], percent_out[1], frame_res, stream);

    auto min = percent_out[0];
    auto scale = 1.0f / (percent_out[1] - min);
    const auto scale_op = [min, scale] __device__(const float pixel) { return (pixel - min) * scale; };

    thrust::transform(exec_policy, gpu_arr, gpu_arr + frame_res, gpu_arr, scale_op);
}

__global__ void kernel_rotate_hsv_to_contiguous_z(
    float* output, const cuComplex* input, const uint frame_res, const uint width, const uint range)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t depth = id / frame_res;
    size_t frame_pos = id % frame_res;
    size_t y = frame_pos / width;
    size_t x = frame_pos % width;
    const size_t rotated_id = (y * range * width) + (x * range) + depth;

    float val = fabsf(input[id].x);
    output[rotated_id] = val;
}

void rotate_hsv_to_contiguous_z(float* output,
                                const cuComplex* input,
                                const uint frame_res,
                                const uint width,
                                const uint range,
                                const cudaStream_t stream)
{
    const uint total_size = frame_res * range;
    cudaSafeCall(cudaMalloc(&output, total_size * sizeof(float)));

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(total_size, threads);

    kernel_rotate_hsv_to_contiguous_z<<<blocks, threads, 0, stream>>>(output, input, frame_res, width, range);
    cudaCheckError();
}

__global__ void
kernel_from_distinct_components_to_interweaved_components(float* output, const float* input, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        output[id * 3] = input[id];
        output[id * 3 + 1] = input[id + frame_res];
        output[id * 3 + 2] = input[id + frame_res * 2];
    }
}

void from_distinct_components_to_interweaved_components(float* output,
                                                        const float* input,
                                                        size_t frame_res,
                                                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_from_distinct_components_to_interweaved_components<<<blocks, threads, 0, stream>>>(output, input, frame_res);
    cudaCheckError();
}

__global__ void
kernel_from_interweaved_components_to_distinct_components(float* output, const float* input, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        output[id] = input[id * 3];
        output[id + frame_res] = input[id * 3 + 1];
        output[id + frame_res * 2] = input[id * 3 + 2];
    }
}

void from_interweaved_components_to_distinct_components(float* output,
                                                        const float* input,
                                                        size_t frame_res,
                                                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_from_interweaved_components_to_distinct_components<<<blocks, threads, 0, stream>>>(output, input, frame_res);
    cudaCheckError();
}
