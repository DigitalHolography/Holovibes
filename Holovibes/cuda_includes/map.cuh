/*! \file
 *
 * \brief Optimized and generic map operation processed gpu side
 *
 * The templated functions can only be called from cuda files (.cu*) files.
 * In order to be exported, the templated function must be declared in .cuh
 * files and implemented in their correponsding .cuhxx files. However, including
 * cuda kernels call .cc files cannot be achieved since those files are not
 * compiled with nvcc.
 *
 * Using a wrapper in .cuhxx and call implemented kernel in .cu does not solve
 * the issue because nvcc compiles .cu* but functions needs to be generate
 * from .cc files.
 * The compilation works fine but no functions are generated from the templates.
 */
#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

// Usings
using uint = unsigned int;
using ushort = unsigned short;

/*! \brief Map input to output throughout a mapping function
 *
 * This function is the generic map operation for any types.
 * It means that it is not the most optimized map operation.
 * For instance, this map operation cannot be vectorized because only
 * some types (float, ushort, uint...) can be vectorized.
 * Moreover, this operation works on any sizes.
 *
 */
template <typename I, typename O, typename FUNC>
void map_generic(const I* const input, O* const output, const size_t size, const FUNC func, const cudaStream_t stream);

/*! \brief Map input (float) to output (float) throughout a mapping function.
 *
 * This function is the specialized map operation for float arrays.
 * It means that it is the most optimized map operation for float arrays.
 * When possible (if size is divisible by four) the vectorized map function is
 * called. Otherwise, the generic (any types, any size) map function is called.
 *
 * This function overloads the templated generic function with float.
 */
template <typename FUNC>
void map_generic(
    const float* const input, float* const output, const size_t size, const FUNC func, const cudaStream_t stream);

/*! \brief Map input (ushort) to output (ushort) throughout a mapping function.
 *
 * \see map_generic float version (above)
 */
template <typename FUNC>
void map_generic(
    const ushort* const input, ushort* const output, const size_t size, const FUNC func, const cudaStream_t stream);

/*! \brief Divide every pixel by a value */
template <typename T>
void map_divide(const T* const input, T* const output, const size_t size, const T value, const cudaStream_t stream);

/*! \brief Multiply every pixel by a value */
template <typename T>
void map_multiply(const T* const input, T* const output, const size_t size, const T value, const cudaStream_t stream);

/* The following functions can be called from any files
 * since they are templated
 */

/*! \brief Apply log10 on every pixel of the input (float array) */
void map_log10(const float* const input, float* const output, const size_t size, const cudaStream_t stream);

/*! \brief Divide every pixel of a float array by a value  */
void map_divide(
    const float* const input, float* const output, const size_t size, const float value, const cudaStream_t stream);

/*! \brief Multiply every pixel of a float array by a value */
void map_multiply(
    const float* const input, float* const output, const size_t size, const float value, const cudaStream_t stream);

// The map.cuhxx file contains functions only compilable via nvcc
// but the cpp compiler still needs the declarations of the templates
#if __NVCC__

#include "map.cuhxx"

#endif
