#include "bw_area.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "hardware_limits.hh"

#define IS_BACKGROUND(VALUE) ((VALUE) == 0.0f)

__global__ void
initialisation_kernel(const float* input, uint* labels, uint* linked, const size_t height, const size_t width)
{
    size_t x = (threadIdx.x + blockIdx.x * blockDim.x);
    size_t y = (threadIdx.y + blockIdx.y * blockDim.y);
    size_t idx = x * width + y;

    if (x < width && y < height)
    {
        labels[idx] = input[idx] * idx;
        linked[idx] = labels[idx];
    }
}

__device__ uint find(uint* linked, uint l)
{
    uint pred = l;
    while (linked[l] != l)
        l = linked[l];
    linked[pred] = l;
    return l;
}

__global__ void
propagate_labels_kernel(uint* labels, uint* linked, const size_t height, const size_t width, size_t* change)
{
    size_t x = (threadIdx.x + blockIdx.x * blockDim.x);
    size_t y = (threadIdx.y + blockIdx.y * blockDim.y);
    size_t idx = x * width + y;

    if (x < width && y < height && labels[x * width + y] != 0)
    {
        int l = labels[idx];

        int n[4];
        n[0] = (y > 0) ? find(linked, labels[idx - 1]) : 0;
        n[1] = (y < width - 1) ? find(linked, labels[idx + 1]) : 0;
        n[2] = (x > 0) ? find(linked, labels[idx - width]) : 0;
        n[3] = (x < height - 1) ? find(linked, labels[idx + width]) : 0;

        int min_l = l;
        for (size_t i = 0; i < 4; i++)
        {
            if (n[i] != 0 && n[i] < min_l)
            {
                linked[min_l] = n[i];
                min_l = n[i];
            }
        }

        if (min_l < l)
        {
            labels[idx] = min_l;
            *change = 1;
        }
    }
}

__global__ void update_label_kernel(uint* labels, uint* linked, const size_t size)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        uint l = labels[idx];
        if (l != linked[idx])
            labels[idx] = find(linked, l);
    }
}

void get_connected_component(uint* labels_d,
                             uint* linked_d,
                             const float* image_d,
                             const size_t width,
                             const size_t height,
                             size_t* change_d,
                             const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width * height, threads);

    initialisation_kernel<<<lblocks, lthreads, 0, stream>>>(image_d, labels_d, linked_d, height, width);
    size_t change_h;
    do
    {
        cudaXMemset(change_d, 0, sizeof(size_t));

        propagate_labels_kernel<<<lblocks, lthreads, 0, stream>>>(labels_d, linked_d, height, width, change_d);
        cudaCheckError();
        cudaXStreamSynchronize(stream);
        cudaXMemcpy(&change_h, change_d, sizeof(size_t), cudaMemcpyDeviceToHost);

        if (change_h)
        {
            update_label_kernel<<<blocks, threads, 0, stream>>>(labels_d, linked_d, width * height);
            cudaCheckError();
        }

    } while (change_h);
    cudaXStreamSynchronize(stream);
}

__global__ void get_labels_sizes_kernel(float* labels_sizes, uint* labels, const size_t size)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size && labels[idx] != 0)
        atomicAdd(labels_sizes + labels[idx], 1);
}

void get_labels_sizes(float* labels_sizes, uint* labels_d, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    cudaXMemset(labels_sizes, 0.0f, size * sizeof(float));

    get_labels_sizes_kernel<<<blocks, threads, 0, stream>>>(labels_sizes, labels_d, size);

    cudaCheckError();
    cudaXStreamSynchronize(stream);
}

__global__ void area_filter_kernel(float* input_output, const uint* label_d, size_t size, uint label_to_keep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        input_output[idx] = (label_d[idx] == label_to_keep) ? 1.0f : 0.0f;
}

void area_filter(float* input_output, const uint* label_d, size_t size, uint label_to_keep, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    area_filter_kernel<<<blocks, threads, 0, stream>>>(input_output, label_d, size, label_to_keep);
    cudaCheckError();
}

__global__ void
area_open_kernel(float* input_output, const uint* label_d, const float* labels_sizes_d, size_t size, uint threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        input_output[idx] = (labels_sizes_d[label_d[idx]] >= threshold) ? 1.0f : 0.0f;
}

void area_open(float* input_output,
               const uint* label_d,
               const float* labels_sizes_d,
               size_t size,
               uint threshold,
               const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    area_open_kernel<<<blocks, threads, 0, stream>>>(input_output, label_d, labels_sizes_d, size, threshold);
    cudaCheckError();
}

void bwareafilt(float* input_output,
                size_t width,
                size_t height,
                uint* labels_d,
                uint* linked_d,
                float* labels_sizes_d,
                size_t* change_d,
                cublasHandle_t& handle,
                cudaStream_t stream)
{
    size_t size = width * height;
    get_connected_component(labels_d, linked_d, input_output, width, height, change_d, stream);

    get_labels_sizes(labels_sizes_d, labels_d, size, stream);

    int maxI = -1;
    cublasIsamax(handle, size, labels_sizes_d, 1, &maxI);
    if (maxI - 1 > 0)
        area_filter(input_output, labels_d, size, maxI - 1, stream);
}

void bwareaopen(float* input_output,
                uint n,
                size_t width,
                size_t height,
                uint* labels_d,
                uint* linked_d,
                float* labels_sizes_d,
                size_t* change_d,
                cudaStream_t stream)
{
    size_t size = width * height;

    get_connected_component(labels_d, linked_d, input_output, width, height, change_d, stream);

    get_labels_sizes(labels_sizes_d, labels_d, size, stream);
    if (n != 0)
        area_open(input_output, labels_d, labels_sizes_d, size, n, stream);
}