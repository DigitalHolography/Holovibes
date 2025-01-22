#include "cuda_memory.cuh"
#include "map.cuh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

static __global__ void
kernel_add_frame_to_sum(float* const input_output, const float* const input, const size_t frame_size)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        input_output[index] += input[index];
}

void add_frame_to_sum(float* const input_output, const float* const input, const size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_add_frame_to_sum<<<blocks, threads, 0, stream>>>(input_output, input, size);
    cudaCheckError();
}

static __global__ void
kernel_subtract_frame_from_sum(float* const input_output, const float* const input, const size_t frame_size)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size)
        input_output[index] -= input[index];
}

void subtract_frame_from_sum(float* const input_output,
                             const float* const input,
                             const size_t size,
                             cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_frame_from_sum<<<blocks, threads, 0, stream>>>(input_output, input, size);
    cudaCheckError();
}

void compute_mean(float* const output,
                  const float* const input,
                  const size_t time_window,
                  const size_t frame_size,
                  cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size, threads);
    map_divide(output, input, frame_size, time_window, stream);
}

__global__ void
kernel_compute_mean_1_2(float* const output, const float* const input, const int frame_size, const int frame_nb_)
{
    extern __shared__ float shared_data[];

    // Each block processes one image, each thread processes one element
    int image_index = blockIdx.x;
    if (image_index >= frame_nb_)
        return;

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

__global__ void kernel_image_centering(
    float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const uint length_video)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size * length_video)
        output[index] = m0_video[index] - m0_mean[index % frame_size]; // Modulo inside kernel is probably unoptimized
}

void image_centering(float* output,
                     const float* m0_video,
                     const float* m0_mean,
                     const size_t frame_size,
                     const size_t length_video,
                     const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size * length_video, threads);

    kernel_image_centering<<<blocks, threads, 0, stream>>>(output, m0_video, m0_mean, frame_size, length_video);
    cudaCheckError();
}