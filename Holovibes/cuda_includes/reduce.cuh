/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once
using uint = unsigned int;

/*! \brief Reduce operation gpu side
 *
 * \param input Input buffer
 * \param result Result of the reduce (even with double, imprecision may arise)
 * \param size Input size
 * \param reduce_op Operator used for the reduction
 * \param atomic_op Atomic operator used for the write back in result
 * \param identity_elt Identity element needed to initilize data (add needs 0, min needs max...)
 *
 * This kernel has been highly tuned in order to maximize the memory bandwidth usage
 * Numerous benches have been done to achieve the best result possible
 * Don't modify this kernel unless making benches
 */
template <typename T, typename U, typename R_OP, typename A_OP>
void reduce_generic(const T* const __restrict__ input,
                    const uint size,
                    const R_OP reduce_op,
                    const A_OP atomic_op,
                    const T identity_elt,
                    const cudaStream_t stream);

/*! \brief Reduce add operation
 *
 * Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
 */
template <typename T, typename U>
void reduce_add(const T* const input, U* const result, const uint size, const cudaStream_t stream);

// Min / Max : ushort not supported by CUDA (because of atomic operation)

/*! \brief Reduce min operation */
template <typename T>
void reduce_min(const T* const input, T* const result, const uint size, const cudaStream_t stream);

/*! \brief Reduce max operation */
template <typename T>
void reduce_max(const T* const input, T* const result, const uint size, const cudaStream_t stream);

#include "reduce.cuhxx"
