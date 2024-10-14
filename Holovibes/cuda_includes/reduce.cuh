/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once
using uint = unsigned int;

/*! \brief Reduce add operation
 *
 * Two types are needed to avoid overflow (sizeof(U) >= sizeof(T))
 */
template <typename T, typename U>
void reduce_add(U* const output, const T* const input, const uint size, const cudaStream_t stream);

// Min / Max : ushort not supported by CUDA (because of atomic operation)

/*! \brief Reduce min operation */
template <typename T>
void reduce_min(T* const output, const T* const input, const uint size, const cudaStream_t stream);

/*! \brief Reduce max operation */
template <typename T>
void reduce_max(T* const output, const T* const input, const uint size, const cudaStream_t stream);

#include "reduce.cuhxx"
