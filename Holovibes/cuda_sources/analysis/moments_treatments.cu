#include "cuda_memory.cuh"

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
    cudaCheckError();
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
    cudaCheckError();
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
    cudaCheckError();
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

void compute_mean_1_2(
    float* const output, const float* const input, const size_t frame_size, const size_t frame_nb, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_nb, threads);
    size_t sharedMemSize = threads / blocks * sizeof(float);
    kernel_compute_mean_1_2<<<blocks, threads, sharedMemSize, stream>>>(output, input, frame_size, frame_nb);
    cudaCheckError();
}

// __global__ void
// kernel_image_centering(float* output, const float* m0_video, const float* m0_mean, const uint frame_size)
// {
//     const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
//     const size_t mean_index = index % 506;
//     if (index < frame_size)
//         output[index] = m0_video[index] - m0_mean[mean_index];
// }

// void image_centering(
//     float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const cudaStream_t stream)
// {
//     uint threads = get_max_threads_1d();
//     uint blocks = map_blocks_to_problem(frame_size * 506, threads);
//     kernel_image_centering<<<blocks, threads, 0, stream>>>(output, m0_video, m0_mean, frame_size * 506);
//     cudaCheckError();
// }

// __global__ void kernel_image_centering(
//     float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const uint mean_size)
// {
//     // Dynamically allocated shared memory
//     extern __shared__ float shared_mean[];

//     // Load mean values into shared memory
//     if (threadIdx.x < mean_size)
//         shared_mean[threadIdx.x] = m0_mean[threadIdx.x];
//     __syncthreads(); // Ensure all threads have loaded shared_mean

//     const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

//     // Ensure we stay within bounds
//     if (index < frame_size)
//     {
//         const size_t mean_index = index % mean_size; // Modulo operation with dynamic mean size
//         output[index] = m0_video[index] - shared_mean[mean_index];
//     }
// }

__global__ void kernel_image_centering(
    float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const uint length_video)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < frame_size * length_video)
        output[index] = m0_video[index] - m0_mean[index % frame_size];
}

void image_centering(float* output,
                     const float* m0_video,
                     const float* m0_mean,
                     const uint frame_size,
                     const uint length_video,
                     const cudaStream_t stream)
{
    // Determine optimal thread count per block
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_size * length_video, threads);

    // Launch the kernel with dynamic shared memory for the mean
    kernel_image_centering<<<blocks, threads, 0, stream>>>(output, m0_video, m0_mean, frame_size, length_video);
    cudaCheckError(); // Check for CUDA errors
}