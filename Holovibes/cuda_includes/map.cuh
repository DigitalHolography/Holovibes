/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file Optimized and generic map operation processed gpu side
 *
 * The templated functions can only be called from cuda files (.cu*) files.
 * In order to be exported, the templated function must be declared in .cuh files
 * and implemented in their correponsding .cuhxx files.
 * However, including cuda kernels call .cc files cannot be achieved since
 * those files are not compiled with nvcc.
 *
 * Using a wrapper in .cuhxx and call implemented kernel in .cu does not solve
 * the issue because nvcc compiles .cu* but functions needs to be generate
 * from .cc files.
 * The compilation works fine but no functions are generated from the templates.
 */
#pragma once

#include <cuda_runtime.h>
#include "Common.cuh"

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
void map_generic(const I* const input,
                 O* const output,
                 const size_t size,
                 const FUNC func,
                 const cudaStream_t stream = 0);

/*! \brief Map input (float) to output (float) throughout a mapping function.
 *
 * This function is the specialized map operation for float arrays.
 * It means that it is the most optimized map operation for float arrays.
 * When possible (if size is divisible by four) the vectorized map function is called.
 * Otherwise, the generic (any types, any size) map function is called.
 *
 * This function overloads the templated generic function with float.
 */
template <typename FUNC>
void map_generic(const float* const input,
                 float* const output,
                 const size_t size,
                 const FUNC func,
                 const cudaStream_t stream = 0);

/*! \brief Map input (ushort) to output (ushort) throughout a mapping function.
 *
 * \see map_generic float version (above)
 *
 */
template <typename FUNC>
void map_generic(const ushort* const input,
                 ushort* const output,
                 const size_t size,
                 const FUNC func,
                 const cudaStream_t stream = 0);

/*! \brief Divide every pixel by a value */
template <typename T>
void map_divide(const T* const input,
                T* const output,
                const size_t size,
                const T value,
                const cudaStream_t stream = 0);

/*! \brief Multiply every pixel by a value */
template <typename T>
void map_multiply(const T* const input,
                  T* const output,
                  const size_t size,
                  const T value,
                  const cudaStream_t stream = 0);

/* The following functions can be called from any files
 * since they are templated
 */

/*! \brief Apply log10 on every pixel of the input (float array) */
void map_log10(const float* const input,
               float* const output,
               const size_t	size,
               const cudaStream_t	stream = 0);

/*! \brief Divide every pixel of a float array by a value  */
void map_divide(const float* const input,
                float* const output,
                const size_t size,
                const float value,
                const cudaStream_t stream = 0);

/*! \brief Multiply every pixel of a float array by a value */
void map_multiply(const float* const input,
                  float* const output,
                  const size_t size,
                  const float value,
                  const cudaStream_t stream = 0);

#include "map.cuhxx"