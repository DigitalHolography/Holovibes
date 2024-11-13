#include "bw_area_filter.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "hardware_limits.hh"

#define IS_BACKGROUND(VALUE) ((VALUE) == 0.0f)

__device__ void lock(uint* mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0)
        ;
}

__device__ void unlock(uint* mutex) { atomicExch(mutex, 0); }

__device__ uint get_linked_label(uint label, uint* linked_d)
{
    size_t pred = label;
    while (label != linked_d[label])
    {
        label = linked_d[label];
    }
    linked_d[pred] = label;
    return label;
}

__global__ void
first_pass_kernel1(const float* image_d, uint* labels_d, uint* linked_d, const size_t width, const size_t height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int x = (idx / width) * 2;
    int y = (idx % width) * 2;
    idx = x * width + y;

    if (y >= 1 && x >= 1 && x < (width - 1) && (y < height - 1) && !IS_BACKGROUND(image_d[idx]))
    {
        linked_d[idx] = idx;
        labels_d[idx] = idx;
    }
}

__device__ void check_and_update_link(
    uint* labels_d, uint* linked_d, const size_t idx, uint* neighbors, const uint nb_neighbors, uint* mutex)
{
    if (nb_neighbors == 0)
    {
        linked_d[idx] = idx;
        labels_d[idx] = idx;
    }
    else
    {

        lock(mutex);
        for (size_t i = 0; i < nb_neighbors; i++)
        {
            neighbors[i] = get_linked_label(labels_d[neighbors[i]], linked_d);
        }

        int min_l = 0;
        for (int k = 1; k < nb_neighbors; k++)
        {
            min_l = neighbors[k] < neighbors[min_l] ? k : min_l;
        }

        size_t label_min = neighbors[min_l];
        labels_d[idx] = label_min;

        for (int k = 1; k < nb_neighbors; k++)
        {
            linked_d[neighbors[(min_l + k) % nb_neighbors]] = label_min;
        }
        unlock(mutex);
    }
}

__global__ void first_pass_kernel2(
    const float* image_d, uint* labels_d, uint* linked_d, const size_t width, const size_t height, uint* mutex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int x = (idx / width) * 2 + 1;
    int y = (idx % width) * 2;
    idx = x * width + y;

    if (y >= 1 && x >= 1 && x < (width - 1) && y < (height - 1) && !IS_BACKGROUND(image_d[idx]))
    {
        uint neighbors[2];
        uint nb_neighbors = 0;

        for (int k = -1; k <= 1; k += 2)
        {
            size_t jdx = (x + k) * width + y;
            if (labels_d[jdx])
                neighbors[nb_neighbors++] = jdx;
        }
        check_and_update_link(labels_d, linked_d, idx, neighbors, nb_neighbors, mutex);
    }
}

__global__ void first_pass_kernel3(
    const float* image_d, uint* labels_d, uint* linked_d, const size_t width, const size_t height, uint* mutex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int x = (idx / width) * 2;
    int y = (idx % width) * 2 + 1;
    idx = x * width + y;

    if (y >= 1 && x >= 1 && x < (width - 1) && y < (height - 1) && !IS_BACKGROUND(image_d[idx]))
    {
        uint neighbors[2];
        uint nb_neighbors = 0;

        for (int k = -1; k <= 1; k += 2)
        {
            size_t jdx = x * width + y + k;
            if (labels_d[jdx])
                neighbors[nb_neighbors++] = jdx;
        }
        check_and_update_link(labels_d, linked_d, idx, neighbors, nb_neighbors, mutex);
    }
}

__global__ void first_pass_kernel4(
    const float* image_d, uint* labels_d, uint* linked_d, const size_t width, const size_t height, uint* mutex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int x = (idx / width) * 2 + 1;
    int y = (idx % width) * 2 + 1;
    idx = x * width + y;

    if (y >= 1 && x >= 1 && x < (width - 1) && y < (height - 1) && !IS_BACKGROUND(image_d[idx]))
    {
        uint neighbors[4];
        uint nb_neighbors = 0;

        for (int k = -1; k <= 1; k += 2)
        {
            size_t jdx = x * width + y + k;
            if (labels_d[jdx])
                neighbors[nb_neighbors++] = jdx;
            jdx = (x + k) * width + y;
            if (labels_d[jdx])
                neighbors[nb_neighbors++] = jdx;
        }

        check_and_update_link(labels_d, linked_d, idx, neighbors, nb_neighbors, mutex);
    }
}

/*!
 * 1 Iterate through each element of the data
 * 2 If the element is not the background
 *      1 Get the neighboring elements of the current element
 *      2 If there are no neighbors, uniquely label the current element and continue
 *      3 Otherwise, find the neighbor with a label and assign it to the current element
 *      4 Store the equivalence between neighboring labels
 *
 *  in linear computation we iterate by column, then by row (Raster Scanning)
 *  but in parallel computation we need to be carefule with memory and i use a methode for checking neighbors who are
 *  already compute so i divide the image in 'pixel' of 4 square pixel :
 *
 *  |1|2|
 *  |3|4|
 *
 *  I run a kernel for each pixel 1 and each pixel 2, and then 3, and for finish pixel 4
 *  each of this kernel is naming first_pass_kernel[1-4]()
 *  and i need also to update some linke of label (equivalence) so i use an homemade cuda mutex for this *
 * */
void first_pass(const float* image_d,
                uint* labels_d,
                uint* linked_d,
                uint* size_t_gpu_,
                const size_t width,
                const size_t height,
                const cudaStream_t stream)
{
    size_t size = width * height;
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size / 2, threads);
    cudaXMemset(linked_d, 0, sizeof(uint));
    cudaXMemset(size_t_gpu_, 0, sizeof(uint));

    first_pass_kernel1<<<blocks, threads, 0, stream>>>(image_d, labels_d, linked_d, width, height);
    first_pass_kernel2<<<blocks, threads, 0, stream>>>(image_d, labels_d, linked_d, width, height, size_t_gpu_);
    first_pass_kernel3<<<blocks, threads, 0, stream>>>(image_d, labels_d, linked_d, width, height, size_t_gpu_);
    first_pass_kernel4<<<blocks, threads, 0, stream>>>(image_d, labels_d, linked_d, width, height, size_t_gpu_);
    cudaCheckError();
}

/*!
 * \brief
 * 1 Iterate through each element of the data
 * 2 If the element is not the background
 *   1 Relabel the element with the equivalent label
 *
 * I use this seconde pass to get the size of each label,
 * the label size count can be in the first pass (i try but does not work)
 * and the second pass is useless if we get the equivalence label during area_filter
 */
__global__ void second_pass_kernel(uint* labels_d, size_t size, uint* linked_d, float* labels_sizes_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && labels_d[idx] != 0)
    {
        uint l = get_linked_label(labels_d[idx], linked_d);
        labels_d[idx] = l;
        atomicAdd(labels_sizes_d + l, 1.0f);
    }
}

void get_connected_component(uint* labels_d,
                             float* labels_sizes_d,
                             uint* linked_d,
                             uint* size_t_gpu_,
                             const float* image_d,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream)
{
    size_t size = width * height;
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    cudaXMemset(labels_d, 0, size * sizeof(uint));
    cudaXMemset(labels_sizes_d, 0, size * sizeof(float));

    first_pass(image_d, labels_d, linked_d, size_t_gpu_, width, height, stream);

    second_pass_kernel<<<blocks, threads, 0, stream>>>(labels_d, size, linked_d, labels_sizes_d);
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