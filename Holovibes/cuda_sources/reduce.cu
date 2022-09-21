#include "cuda.h"
#include "cuda_runtime.h"

#include <cassert>

#include "common.cuh"
#include "cuda_memory.cuh"
#include "reduce.cuh"

/*! Mask used by the warp reduce using registers.
 * This mask means every thread should be processed
 */
#define FULL_MASK 0xffffffff
/*! \brief Number of threads handle by an unique warp */
static constexpr uint warp_size = 32;
/*! \brief Maximum number of threads in one block */
static constexpr uint max_block_size = 1024;

/*! \brief Reduce warp kernel
 *
 * Differently from the reference, we are not reducing in shared memory
 * but via threads register
 */
template <typename T, unsigned int pixel_per_block>
__device__ void warp_primitive_reduce(T* sdata, const uint tid)
{
    /* A thread warp (32) works on twice its size
     * __shfl_down_sync instruction use thread register to reduce
     * In our case each thread hold one register
     *
     * This first step needs to be done in shared memory because
     * the pixel located further than warp size can't be stored in register
     *
     * This function can work on less than 64 pixels
     * Because the shared data is initialized to 0 for the missing pixels
     * Executing useless += is more performant than branching with multiple if
     */
    T acc_reg = sdata[tid] + sdata[tid + warp_size];

    if (pixel_per_block >= warp_size)
        acc_reg += __shfl_down_sync(FULL_MASK, acc_reg, warp_size / 2);

    if (pixel_per_block >= warp_size / 2)
        acc_reg += __shfl_down_sync(FULL_MASK, acc_reg, warp_size / 4);

    if (pixel_per_block >= warp_size / 4)
        acc_reg += __shfl_down_sync(FULL_MASK, acc_reg, warp_size / 8);

    if (pixel_per_block >= warp_size / 8)
        acc_reg += __shfl_down_sync(FULL_MASK, acc_reg, warp_size / 16);

    if (pixel_per_block >= warp_size / 16)
        acc_reg += __shfl_down_sync(FULL_MASK, acc_reg, warp_size / 32);

    if (tid == 0)
        sdata[tid] = acc_reg;
}

/*! \brief Reduce kernel
 *
 * The reference code of this kernel can be found here :
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * We highly recommand developers that wants to modify this kernel
 * to first checkout and fully understand the above reference
 *
 * We slightly improved performances by tuning it
 */
template <typename T, typename U, uint pixel_per_block>
__global__ void kernel_reduce(const T* const __restrict__ input, U* const __restrict__ result, const uint size)
{
    // Each block is reduced in shared data (avoiding multiple global memory
    // acceses)
    extern __shared__ T sdata[];
    const uint tid = threadIdx.x;

    sdata[tid] = static_cast<T>(0);
    /* Scope to apply stride loop design pattern
    ** The goal is to have a flexible kernel that can handle input sizes bigger
    *than the grid
    ** This increases threads work and thus improves performances
    */
    {
        const uint problem_stride = pixel_per_block * gridDim.x;

        // Accumulate in shared memory the first reduction step on multiple
        // frame blocks
        for (uint i = blockIdx.x * pixel_per_block + tid;; i += problem_stride)
        {
            if (i + (pixel_per_block / 2) < size)
                sdata[tid] += input[i] + input[i + (pixel_per_block / 2)];
            else if (i < size)
                sdata[tid] += input[i];
            else
                break;
        }
    }

    // Wait for all threads of the block to finish accumulation
    __syncthreads();

    // At this step, the number of pixel per block has been halved by the first
    // reduction step

    // Thread block reduction
    if (pixel_per_block / 2 >= max_block_size)
    {
        if (tid < max_block_size / 2)
            sdata[tid] += sdata[tid + max_block_size / 2];
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 2)
    {
        if (tid < max_block_size / 4)
            sdata[tid] += sdata[tid + max_block_size / 4];
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 4)
    {
        if (tid < max_block_size / 8)
            sdata[tid] += sdata[tid + max_block_size / 8];
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 8)
    {
        if (tid < max_block_size / 16)
            sdata[tid] += sdata[tid + max_block_size / 16];
        __syncthreads();
    }

    // The last reductions steps is a warp size problem and can thus be
    // optimized
    if (tid < warp_size)
        warp_primitive_reduce<T, pixel_per_block / 2>(sdata, tid);

    // Each block writes it local reduce to global memory
    if (tid == 0)
        atomicAdd(result, sdata[tid]);
}

void gpu_reduce(const float* const input, double* const result, const uint size, const cudaStream_t stream)
{
    // Most optimized grid layout for Holovibes input sizes
    constexpr uint optimal_nb_blocks = 1024;
    constexpr uint optimal_block_size = 128;
    // Handling block_size smaller than 64 would reduce warp_reduce performances
    CHECK(optimal_block_size >= 64,
          "kernel reduce only works with with a block size equal or greater than 64 threads (128 pixels)");

    // We still reduce the number of blocks if this reduce is used for really
    // small input
    const uint nb_blocks = std::min((size - 1) / (optimal_block_size * 2) + 1, optimal_nb_blocks);

    // Reset result to 0
    cudaXMemsetAsync(result, 0, sizeof(double), stream);

    // Each thread works at least on 2 pixels
    kernel_reduce<float, double, optimal_block_size * 2>
        <<<nb_blocks, optimal_block_size, optimal_block_size * sizeof(float), 0, stream>>>(input, result, size);
    cudaCheckError();
}
