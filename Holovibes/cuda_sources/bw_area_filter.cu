#include "bw_area_filter.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "hardware_limits.hh"

using uint = unsigned int;

#define IS_BACKGROUND(VALUE) ((VALUE) == 1.0f)

struct vector_t
{
    size_t* ptr_d;
    size_t size;
    size_t max_size;
};

__device__ void add_label_link(vector_t& linked)
{
    if (linked.size >= linked.max_size)
    {
        size_t* new_ptr;
        cudaXMalloc(&new_ptr, linked.max_size * 2 * sizeof(size_t));
        cudaXMemcpy(new_ptr, linked.ptr_d, linked.max_size * sizeof(size_t));
        cudaXFree(linked.ptr_d);
        linked.ptr_d = new_ptr;
        linked.max_size *= 2;
    }
    linked.ptr_d[linked.size] = linked.size++;
}

__global__ void
first_pass(const float* image_d, size_t* labels_d, vector_t& linked, const size_t width, const size_t height)
{
    for (size_t i = 1; i < width - 1; i++)
    {
        for (size_t j = 1; j < height - 1; j++)
        {
            if (!IS_BACKGROUND(image_d[i * width + j]))
            {
                size_t neighbors[4][2]; // 4 neighbors
                size_t nb_neighbors = 0;

                if (!IS_BACKGROUND(image_d[(i - 1) * width + j]))
                {
                    neighbors[nb_neighbors][0] = i - 1;
                    neighbors[nb_neighbors++][1] = j;
                }
                if (!IS_BACKGROUND(image_d[i * width + (j - 1)]))
                {
                    neighbors[nb_neighbors][0] = i - 1;
                    neighbors[nb_neighbors++][1] = j;
                }

                if (nb_neighbors == 0)
                {
                    add_label_link(linked);
                    labels_d[i * width + j] = linked.size - 1;
                }
                else if (nb_neighbors == 1)
                {
                    labels_d[i * width + j] = labels_d[neighbors[0][0] * width + neighbors[0][1]];
                }
                else
                {
                    // Find the smallest label and update label link

                    size_t L[4]; // 4 neighbors
                    size_t nb_neighbors_labels = 0;
                    labels_d[i * width + j] = labels_d[neighbors[0][0] * width + neighbors[0][1]];
                    for (size_t p = 1; p < nb_neighbors; p++)
                    {
                        labels_d[i * width + j] =
                            std::min(labels_d[i * width + j], labels_d[neighbors[p][0] * width + neighbors[p][1]]);
                    }
                    for (size_t p = 0; p < nb_neighbors; p++)
                    {
                        size_t l = labels_d[neighbors[p][0] * width + neighbors[p][1]];
                        linked.ptr_d[l] = labels_d[i * width + j] < l ? labels_d[i * width + j] : l;
                    }
                }
            }
        }
    }
}

__global__ void second_pass_kernel(size_t* labels_d, size_t size, size_t* linked_d, size_t* labels_sizes_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && labels_d[idx] != 0)
    {
        labels_d[idx] = linked_d[labels_d[idx]];
        atomicAdd(labels_sizes_d + labels_d[idx], 1);
    }
}

#define DEFAULT_SIZE_LINKED_D 2048

void get_connected_component(const float* image_d,
                             size_t* labels_d,
                             size_t** labels_sizes_d,
                             size_t& nb_labels,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream)
{
    size_t size = width * height;
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    vector_t linked = {nullptr, 0, DEFAULT_SIZE_LINKED_D};
    cudaXMalloc(&linked.ptr_d, DEFAULT_SIZE_LINKED_D * sizeof(size_t));
    add_label_link(linked); // label 0 is background

    //  First pass
    first_pass(image_d, labels_d, linked, width, height);

    cudaXMalloc(labels_sizes_d, linked.size * sizeof(size_t));
    cudaXMemset(labels_sizes_d, 0, linked.size);
    second_pass_kernel<<<blocks, threads, 0, stream>>>(labels_d, size, linked.ptr_d, *labels_sizes_d);
    cudaDeviceSynchronize();
}

__device__ __host__ void swap(size_t* T, size_t i, size_t j)
{
    size_t tmp = T[i];
    T[i] = T[j];
    T[j] = tmp;
}

__device__ __host__ void get_n_max_index(size_t* labels_size_d, size_t nb_label, size_t* labels_max_d, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        size_t j = i;
        labels_max_d[j] = j;
        while (j > 0 && labels_size_d[labels_max_d[j - 1]] > labels_size_d[labels_max_d[j]])
        {
            swap(labels_max_d, j, j - 1);
            j--;
        }
    }
    for (size_t i = n; i < nb_label; i++)
    {
        if (labels_size_d[i] > labels_size_d[labels_max_d[0]])
        {
            labels_max_d[0] = i;
            size_t j = 1;
            while (j < n && labels_size_d[labels_max_d[j - 1]] > labels_size_d[labels_max_d[j]])
            {
                swap(labels_max_d, j, j - 1);
                j++;
            }
        }
    }
}

__global__ void area_filter_kernel(float* image_d, const size_t* label_d, size_t size, int* is_keep_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        image_d[idx] = is_keep_d[label_d[idx]] ? 0.0f : 1.0f;
}

void area_filter(float* image_d, const size_t* label_d, size_t size, int* is_keep_d, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    area_filter_kernel<<<blocks, threads, 0, stream>>>(image_d, label_d, size, is_keep_d);
    cudaDeviceSynchronize();
}

__global__ void create_is_keep_in_label_size(size_t* labels_sizes_d, size_t nb_labels, size_t* labels_max_d, size_t n)
{
    cudaXMemset(labels_sizes_d, 0, nb_labels * sizeof(size_t));
    for (size_t i = 0; i < n; i++) // TODO use a kernel
        labels_sizes_d[i] = 1;
}
