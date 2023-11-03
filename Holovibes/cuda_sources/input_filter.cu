#include "input_filter.cuh"

#include "cuda_memory.cuh"
#include "shift_corners.cuh"

void __global__ kernel_multiply_filters(float* gpu_filter2d_mask, float* gpu_input_filter, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        gpu_filter2d_mask[id] *= gpu_input_filter[id];
    }
}

void apply_filter(float* gpu_filter2d_mask,
                  float* gpu_input_filter_mask,
                  const float* input_filter,
                  size_t width,
                  size_t height,
                  const cudaStream_t stream)
{
    size_t frame_res = width * height;

    cudaXMemcpyAsync(gpu_input_filter_mask, input_filter, frame_res * sizeof(float), cudaMemcpyHostToDevice, stream);

    shift_corners(gpu_input_filter_mask, 1, width, height, stream);

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_multiply_filters<<<blocks, threads, 0, stream>>>(gpu_filter2d_mask, gpu_input_filter_mask, frame_res);
}