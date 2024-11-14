#include "bw_area.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "hardware_limits.hh"

#define IS_BACKGROUND(VALUE) ((VALUE) == 0.0f)

__global__ void init_labels_kernel(uint* labels_d, const float* image_d, const size_t width, const size_t height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx / width;
    int y = idx % width;

    if ((x < width) && (y < height))
    {
        const float value = image_d[x * width + y];

        const bool nn = (y > 0) ? (value == image_d[x * width + (y - 1)]) : false;
        const bool ne = (x > 0) ? (value == image_d[(x - 1) * width + y]) : false;
        const bool nne = ((y > 0) && (x > 0)) ? (value == image_d[(x - 1) * width + (y - 1)]) : false;
        const bool nnw = ((y > 0) && (x < width - 1)) ? (value == image_d[(x + 1) * width + (y - 1)]) : false;

        uint label = (ne) ? ((x - 1) * width + y) : (x * width + y);
        label = (nne) ? (x - 1) * width + (y - 1) : label;
        label = (nn) ? (x * width + (y - 1)) : label;
        label = (nnw) ? (x + 1) * width + (y - 1) : label;

        labels_d[x * width + y] = label;
    }
}
__device__ __inline__ uint find_root(uint* labels, uint label)
{
    uint next = labels[label];

    while (label != next)
    {
        label = next;
        next = labels[label];
    }

    return (label);
}

__global__ void resolve_labels_kernel(uint* labels_d, const size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        labels_d[idx] = find_root(labels_d, labels_d[idx]);
}

__device__ __inline__ uint reduction(uint* labels_d, uint l1, uint l2)
{
    uint tmp = (l1 != l2) ? labels_d[l1] : 0;

    while ((l1 != l2) && (l1 != tmp))
    {
        l1 = tmp;
        tmp = labels_d[l1];
    }
    tmp = (l1 != l2) ? labels_d[l2] : 0;

    while ((l1 != l2) && (l2 != tmp))
    {
        l2 = tmp;
        tmp = labels_d[l2];
    }

    uint l3;
    while (l1 != l2)
    {
        if (l1 < l2)
        {
            l1 = l1 ^ l2;
            l2 = l1 ^ l2;
            l1 = l1 ^ l2;
        }

        l3 = atomicMin(&labels_d[l1], l2);
        l1 = (l1 == l3) ? l2 : l3;
    }

    return l1;
}

__global__ void label_reduction_kernel(uint* labels_d, const float* image_d, const size_t width, const size_t height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx / width;
    int y = idx % width;

    if ((x < width) && (y < height))
    {
        const float value = image_d[x * width + y];
        const bool nn = (y > 0) ? (value == image_d[x * width + (y - 1)]) : false;

        if (!nn)
        {
            const bool nne = ((y > 0) && (x > 0)) ? (value == image_d[(x - 1) * width + y - 1]) : false;
            const bool ne = (x > 0) ? (value == image_d[(x - 1) * width + y]) : false;
            const bool nw = ((y > 0) && (x < width - 1)) ? (value == image_d[(x + 1) * width + y - 1]) : false;

            if (nw)
            {
                if ((nne && ne) || (nne && !ne))
                    reduction(labels_d, labels_d[x * width + y], labels_d[(x + 1) * width + y - 1]);

                if (!nne && ne)
                    reduction(labels_d, labels_d[x * width + y], labels_d[(x - 1) * width + y]);
            }
        }
    }
}

__global__ void resolve_background(uint* labels_d, const float* image_d, float* labels_sizes_d, const size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        labels_d[idx] = (!IS_BACKGROUND(image_d[idx])) ? labels_d[idx] + 1 : 0;
        if (labels_d[idx])
            atomicAdd(labels_sizes_d + labels_d[idx], 1.0f);
    }
}

void get_connected_component(uint* labels_d,
                             float* labels_sizes_d,
                             uint* linked_d,
                             const float* image_d,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream)
{
    size_t size = width * height;
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    cudaXMemset(labels_sizes_d, 0, size * sizeof(float));

    init_labels_kernel<<<blocks, threads, 0, stream>>>(labels_d, image_d, width, height);
    resolve_labels_kernel<<<blocks, threads, 0, stream>>>(labels_d, size);

    label_reduction_kernel<<<blocks, threads, 0, stream>>>(labels_d, image_d, width, height);
    resolve_labels_kernel<<<blocks, threads, 0, stream>>>(labels_d, size);

    resolve_background<<<blocks, threads, 0, stream>>>(labels_d, image_d, labels_sizes_d, size);

    cudaCheckError();
    cudaDeviceSynchronize();
}

__global__ void area_filter_kernel(float* image_d, const uint* label_d, size_t size, uint label_to_keep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        image_d[idx] = (label_d[idx] == label_to_keep) ? 1.0f : 0.0f;
}

void area_filter(float* image_d, const uint* label_d, size_t size, uint label_to_keep, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    area_filter_kernel<<<blocks, threads, 0, stream>>>(image_d, label_d, size, label_to_keep);
    cudaCheckError();
}

__global__ void area_open_kernel(float* image_d, const uint* label_d, const float* labels_sizes_d, size_t size, uint p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        image_d[idx] = (labels_sizes_d[label_d[idx]] >= p) ? 1.0f : 0.0f;
}

void area_open(
    float* image_d, const uint* label_d, const float* labels_sizes_d, size_t size, uint p, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    area_open_kernel<<<blocks, threads, 0, stream>>>(image_d, label_d, labels_sizes_d, size, p);
    cudaCheckError();
}