#include "tools_analysis.cuh"
#include "cuda_memory.cuh"
#include "tools_analysis_debug.hh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define CIRCLE_MASK_RADIUS 0.07f

// __global__ void kernel_compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth)
// {
//     const uint index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (index < depth * size)
//     {
//         const uint depth_index = index / size;
//         const uint size_index = index % size;

//         atomicAdd(&output[depth_index], A[size_index + depth_index * size] * B[size_index]);
//     }
// }

// __global__ void kernel_divide(float* output, size_t depth, size_t size)
// {
//     const uint index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < depth)
//         output[index] /= size;
// }

__global__ void kernel_compute_multiplication_mean(float* output, float* A, float* B, size_t size, uint depth)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < depth)
    {
        for (uint i = 0; i < size; i++)
            output[index] += A[i + index * size] * B[i];
        output[index] /= size;
    }
}

__global__ void kernel_compute_multiplication_mean_optimized(
    float* output, const float* A, const float* B, size_t size, uint depth, size_t i)
{
    // Création de la mémoire partagée pour la réduction locale
    extern __shared__ float sdata[];
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint thread_id = threadIdx.x;

    // Initialisation locale des résultats
    float temp = 0.0f;

    // Accéder aux données de manière coalescente si possible
    if (index < size)
    {
        temp = A[index + i * size] * B[index];
    }

    // Réduction parallèle dans la mémoire partagée
    sdata[thread_id] = temp;
    __syncthreads();

    // Réduction en utilisant l'arbre binaire
    for (uint s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (thread_id < s)
        {
            sdata[thread_id] += sdata[thread_id + s];
        }
        __syncthreads();
    }

    // Le thread 0 écrit le résultat partiel
    if (thread_id == 0)
    {
        atomicAdd(output + i, sdata[0]);
    }
}

__global__ void kernel_divide(float* output, size_t denominator, uint depth)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < depth)
        output[index] /= denominator;
}

void compute_multiplication_mean(float* output, float* A, float* B, size_t size, uint depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    for (size_t i = 0; i < depth; ++i)
    {
        size_t shared_mem_size = threads * sizeof(float);
        kernel_compute_multiplication_mean_optimized<<<blocks, threads, shared_mem_size, stream>>>(output,
                                                                                                   A,
                                                                                                   B,
                                                                                                   size,
                                                                                                   depth,
                                                                                                   i);
        cudaCheckError();
    }
    blocks = map_blocks_to_problem(depth, threads);
    kernel_divide<<<blocks, threads, 0, stream>>>(output, size, depth);
    cudaCheckError();
}

// void compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream)
// {
//     uint threads = get_max_threads_1d();
//     uint blocks = map_blocks_to_problem(depth * size, threads);
//     kernel_compute_multiplication_mean<<<blocks, threads, 0, stream>>>(output, A, B, size, depth);
//     cudaCheckError();

//     threads = get_max_threads_1d();
//     blocks = map_blocks_to_problem(depth, threads);
//     kernel_divide<<<blocks, threads, 0, stream>>>(output, depth, size);
//     cudaCheckError();
// }

__global__ void kernel_compute_multiplication(float* output, float* A, float* B, size_t size, size_t depth)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        for (uint i = 0; i < depth; i++)
            output[i * size + index] = A[i * size + index] * B[index];
    }
}

void compute_multiplication(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_compute_multiplication<<<blocks, threads, 0, stream>>>(output, A, B, size, depth);
    cudaCheckError();
}

int find_max_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
    return max_ptr - dev_ptr;
}

int find_min_thrust(float* input, size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + size);
    return min_ptr - dev_ptr;
}

int compute_barycentre_circle_mask(float* output,
                                   float* crv_circle_mask,
                                   float* input,
                                   size_t width,
                                   size_t height,
                                   cudaStream_t stream,
                                   int CRV_index)
{
    size_t size = width * height;
    int barycentre_index = find_max_thrust(input, size);
    if (CRV_index == -1)
        CRV_index = find_min_thrust(input, size);

    compute_circle_mask(output,
                        barycentre_index % width,
                        std::floor(barycentre_index / height),
                        CIRCLE_MASK_RADIUS * (width + height) / 2,
                        width,
                        height,
                        stream);

    compute_circle_mask(crv_circle_mask,
                        CRV_index % width,
                        std::floor(CRV_index / height),
                        CIRCLE_MASK_RADIUS * (width + height) / 2,
                        width,
                        height,
                        stream);

    apply_mask_or(output, crv_circle_mask, width, height, stream);

    return CRV_index;
}