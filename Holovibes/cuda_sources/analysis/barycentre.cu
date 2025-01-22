#include "tools_analysis.cuh"
#include "cuda_memory.cuh"
#include "tools_analysis_debug.hh"
#include "map.cuh"

__global__ void
kernel_compute_multiplication_mean(float* output, const float* A, const float* B, size_t size, uint depth, size_t i)
{
    extern __shared__ float sdata[];
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint thread_id = threadIdx.x;

    float temp = 0.0f;

    if (index < size)
        temp = A[index + i * size] * B[index];

    sdata[thread_id] = temp;
    __syncthreads();

    for (uint s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (thread_id < s)
            sdata[thread_id] += sdata[thread_id + s];
        __syncthreads();
    }

    if (thread_id == 0)
        atomicAdd(output + i, sdata[0]);
}

void compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    for (size_t i = 0; i < depth; ++i)
    {
        size_t shared_mem_size = threads * sizeof(float);
        kernel_compute_multiplication_mean<<<blocks, threads, shared_mem_size, stream>>>(output, A, B, size, depth, i);
        cudaCheckError();
    }
    blocks = map_blocks_to_problem(depth, threads);
    map_divide(output, depth, size, stream);
    cudaCheckError();
}

int compute_barycentre_circle_mask(float* output,
                                   float* crv_circle_mask,
                                   float* input,
                                   size_t width,
                                   size_t height,
                                   float barycentre_factor,
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
                        barycentre_factor * (width + height) / 2,
                        width,
                        height,
                        stream);

    compute_circle_mask(crv_circle_mask,
                        CRV_index % width,
                        std::floor(CRV_index / height),
                        barycentre_factor * (width + height) / 2,
                        width,
                        height,
                        stream);

    apply_mask_or(output, crv_circle_mask, width, height, stream);

    return CRV_index;
}
