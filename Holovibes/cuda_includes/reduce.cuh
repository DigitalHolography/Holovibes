/*! \file reduce.cuh
 *
 *  \brief Generic functions to apply optimized reduction operations on GPU side. Particuarly useful for min, max,
 *  add,.. or any of these other kind of operations.
 */
#pragma once
using uint = unsigned int;

#include "cuda.h"
#include "cuda_runtime.h"

#include <cassert>
#include <limits>

#include "common.cuh"
#include "cuda_memory.cuh"

/*! \brief Mask used by the warp reduce using registers. This mask means every thread should be processed. */
#define FULL_MASK 0xffffffff

/*! \brief Maximum number of threads in one block. */
static constexpr uint max_block_size = 1024;

/*! \brief Set single device value.
 *
 *  \param[out] output The ptr to store the value.
 *  \param[in] value The templated value to store in the `output` pointer.
 */
template <typename T>
__global__ static void set_cuda_value(T* const output, const T value)
{
    *output = value;
}

/*! \brief Reduce warp kernel.
 *
 *  Use warp-level primitives to reduce in threads register.
 *
 *  A thread warp (32) works on twice its size.
 *  The `__shfl_down_sync` instruction is used to reduce in thread register. In our case each thread hold one register.
 *  This function can work on less than 64 pixels because the shared data is initialized to 0 for the missing pixels.
 *
 *  \param[in out] sdata The shared memory to get the data. The result is stored at index 0 at the end of
 *  parallelisation.
 *  \param[in] tid The current thread index.
 *  \param[in] operation The operation to apply inside the warp. The operation could be: add, min, max,...
 */
template <typename T, unsigned int pixel_per_block, typename OP>
__device__ static void warp_reduce(T* sdata, const uint tid, const OP operation)
{
    // Initial reduction, avoid out-of-bound memory access.
    // This first step needs to be done in shared memory because the pixel located further than warp size can't be
    // stored in register.
    T acc_reg = (tid + warpSize < pixel_per_block) ? operation(sdata[tid], sdata[tid + warpSize]) : sdata[tid];

    // Intra-warp reduction.
    for (int offset = warpSize / 2; offset >= 1; offset /= 2)
    {
        acc_reg = operation(acc_reg, __shfl_down_sync(FULL_MASK, acc_reg, offset));
    }

    // Store result in thread 0.
    if (tid == 0)
        sdata[tid] = acc_reg;
}

/*! \brief Reduce kernel
 *
 *  The reference code of this kernel can be found here :
 *  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *  We highly recommand developers that wants to modify this kernel
 *  to first checkout and fully understand the above reference
 *
 *  We slightly improved performances by tuning it.
 *
 *  \param[out] output The output buffer after hierarchic reduction.
 *  \param[in] input The input to get the values to compute.
 *  \param[in] size The size of the buffer.
 *  \param[in] reduce_op The operation to reduce, e.g: min, max, add,...
 *  \param[in] atomic_op The operation to write result in the output buffer, e.g: atomicAdd, atomicOr,...
 */
template <typename O, typename I, uint pixel_per_block, typename R_OP, typename A_OP>
__global__ static void kernel_reduce(O* const __restrict__ output,
                                     const I* const __restrict__ input,
                                     const uint size,
                                     const R_OP reduce_op,
                                     const A_OP atomic_op,
                                     const I identity_elt)
{
    // Each block is reduced in shared data (avoiding multiple global memory acceses).
    __shared__ I sdata[pixel_per_block / 2];
    const uint tid = threadIdx.x;

    sdata[tid] = identity_elt;
    /* Scope to apply stride loop design pattern.
     * The goal is to have a flexible kernel that can handle input sizes bigger than the grid.
     * This increases threads work and thus improves performances.
     */
    {
        const uint problem_stride = pixel_per_block * gridDim.x;

        // Accumulate in shared memory the first reduction step on multiple frame blocks.
        for (uint i = blockIdx.x * pixel_per_block + tid;; i += problem_stride)
        {
            if (i + (pixel_per_block / 2) < size)
                sdata[tid] = reduce_op(sdata[tid], reduce_op(input[i], input[i + (pixel_per_block / 2)]));
            else if (i < size)
                sdata[tid] = reduce_op(sdata[tid], input[i]);
            else
                break;
        }
    }

    // Wait for all threads of the block to finish accumulation.
    __syncthreads();

    // At this step, the number of pixel per block has been halved by the first reduction step.

    // Thread block reduction.
    if (pixel_per_block / 2 >= max_block_size)
    {
        if (tid < max_block_size / 2)
            sdata[tid] = reduce_op(sdata[tid], sdata[tid + max_block_size / 2]);
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 2)
    {
        if (tid < max_block_size / 4)
            sdata[tid] = reduce_op(sdata[tid], sdata[tid + max_block_size / 4]);
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 4)
    {
        if (tid < max_block_size / 8)
            sdata[tid] = reduce_op(sdata[tid], sdata[tid + max_block_size / 8]);
        __syncthreads();
    }
    if (pixel_per_block / 2 >= max_block_size / 8)
    {
        if (tid < max_block_size / 16)
            sdata[tid] = reduce_op(sdata[tid], sdata[tid + max_block_size / 16]);
        __syncthreads();
    }

    // The last reductions steps is a warp size problem and can thus be optimized.
    if (tid < warpSize)
        warp_reduce<I, pixel_per_block / 2>(sdata, tid, reduce_op);

    // Each block writes it local reduce to global memory.
    if (tid == 0)
        atomic_op(output, sdata[tid]);
}

/*! \brief Reduce operation gpu side.
 *  This kernel has been highly tuned in order to maximize the memory bandwidth usage.
 *  Numerous benches have been done to achieve the best output possible.
 *  Don't modify this kernel unless making benches.
 *
 *  \param[out] output Output of the reduce (even with double, imprecision may arise).
 *  \param[in] input Input buffer.
 *  \param[in] size Input size.
 *  \param[in] reduce_op Operator used for the reduction.
 *  \param[in] atomic_op Atomic operator used for the write back in output.
 *  \param[in] identity_elt Identity element needed to initilize data (add needs 0, min needs max...).
 */
template <typename O, typename I, typename R_OP, typename A_OP>
void reduce_generic(O* const __restrict__ output,
                    const I* const __restrict__ input,
                    const uint size,
                    const R_OP reduce_op,
                    const A_OP atomic_op,
                    const I identity_elt,
                    const cudaStream_t stream)
{
    // Most optimized grid layout for Holovibes input sizes.
    constexpr uint optimal_nb_blocks = 1024;
    constexpr uint optimal_block_size = 128;
    // Handling block_size smaller than 64 would reduce warp_reduce performances.
    CHECK(optimal_block_size >= 64);

    // We still reduce the number of blocks if this reduce is used for really small input.
    const uint nb_blocks = std::min((size - 1) / (optimal_block_size * 2) + 1, optimal_nb_blocks);

    /* Reset output to identity element.
     * cudaMemset can only set a value for each byte (can't memset to 0xFFFFFFF because of float).
     * cudaMemcpyHostToDevice is too slow.
     * The fastest & generic option is using a cuda kernel (~2us).
     */
    set_cuda_value<<<1, 1, 0, stream>>>(output, static_cast<O>(identity_elt));

    // Each thread works at least on 2 pixels.
    kernel_reduce<O, I, optimal_block_size * 2>
        <<<nb_blocks, optimal_block_size, 0, stream>>>(output, input, size, reduce_op, atomic_op, identity_elt);
    cudaCheckError();
}

/*! \brief Reduce add operation.
 *
 *  Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
 *
 *  \param[out] output The output buffer after add reduction.
 *  \param[in] input The input to get the values to compute.
 *  \param[in] size The size of the buffer.
 *  \param[in] stream The CUDA stream to parallelize.
 */
template <typename O, typename I>
void reduce_add(O* const output, const I* const input, const uint size, const cudaStream_t stream)
{
    static const auto op_add = [] __device__(const I left, const I right) { return left + right; };
    static const auto atomic_add = [] __device__(O* const left, const I right) { atomicAdd(left, right); };
    static constexpr I identity_elt_add = static_cast<I>(0);

    reduce_generic(output, input, size, op_add, atomic_add, identity_elt_add, stream);
}

/*! \brief Reduce min operation.
 *
 *  Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
 *  ushort not supported by CUDA (because of atomic operation).
 *
 *  \param[out] output The output buffer after add reduction.
 *  \param[in] input The input to get the values to compute.
 *  \param[in] size The size of the buffer.
 *  \param[in] stream The CUDA stream to parallelize.
 */
template <typename T>
void reduce_min(T* const output, const T* const input, const uint size, const cudaStream_t stream)
{
    static const auto op_min = [] __device__(const T left, const T right) { return min(left, right); };
    static const auto atomic_min = [] __device__(T* const left, const T right) { atomicMin(left, right); };
    // The identity element of min operation is the max of the type : min(A, max(T)) = A.
    static constexpr T identity_elt_min = std::numeric_limits<T>::max();

    reduce_generic(output, input, size, op_min, atomic_min, identity_elt_min, stream);
}

/*! \brief Reduce max operation.
 *
 *  Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
 *  ushort not supported by CUDA (because of atomic operation).
 *
 *  \param[out] output The output buffer after add reduction.
 *  \param[in] input The input to get the values to compute.
 *  \param[in] size The size of the buffer.
 *  \param[in] stream The CUDA stream to parallelize.
 */
template <typename T>
void reduce_max(T* const output, const T* const input, const uint size, const cudaStream_t stream)
{
    static const auto op_max = [] __device__(const T left, const T right) { return max(left, right); };
    static const auto atomic_max = [] __device__(T* const left, const T right) { atomicMax(left, right); };
    // The identity element of max operation is the lowest of the type : max(A, lowest(T)) = A
    // std::numeric_limits<T>::min must not be used.
    // For floating numbers, min != lowest, lowest = -max.
    // From the cpp reference, the min value of floating numbers is the closest value to 0.
    static constexpr T identity_elt_max = std::numeric_limits<T>::lowest();

    reduce_generic(output, input, size, op_max, atomic_max, identity_elt_max, stream);
}