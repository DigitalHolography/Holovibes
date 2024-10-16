/*!
 * @file apply_mask.cuh
 *
 * @brief Contains functions for applying a mask to a number of frames in CUDA.
 *
 * Usage:
 * - To apply a mask to an input buffer in-place, use \ref `apply_mask` without output.
 * - To apply a mask to an input buffer and store the result in an output buffer, use \ref `apply_mask` with an output.
 * - The functions take a CUDA stream as a parameter, which can be used to launch the operation asynchronously.
 *
 * Code example:
 * ```cpp
 * // Create a mask and input buffer
 * cuComplex mask[SIZE];
 * cuComplex input[SIZE * BATCH_SIZE];
 *
 * // Apply the mask to the input buffer in-place
 * apply_mask(input, mask, SIZE, BATCH_SIZE, stream);
 *
 * // Create an output buffer and apply the mask to the input buffer
 * cuComplex output[SIZE * BATCH_SIZE];
 * apply_mask(input, mask, output, SIZE, BATCH_SIZE, stream);
 * ```
 */

#pragma once

#include "cuComplex.h"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief These functions apply a mask to a number of frames in-place
 *
 * \param[in,out] in_out The buffer of images to modify
 * \param[in] mask The mask to apply to 'in_out'
 * \param[in] size The number of pixels in one frame of 'in_out'
 * \param[in] batch_size The number of frames of 'in_out'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(
    cuComplex* __restrict__ in_out, const cuComplex* __restrict__ mask, const size_t size, const uint batch_size, const cudaStream_t stream);

void apply_mask(
    cuComplex* __restrict__ in_out, const float* __restrict__ mask, const size_t size, const uint batch_size, const cudaStream_t stream);

void apply_mask(float* __restrict__ in_out, const float* __restrict__ mask, const size_t size, const uint batch_size, const cudaStream_t stream);

/*! \brief These functions apply a mask to a number of frames and store the result in an output buffer
 *
 * \param[in,out] input The buffer of images to modify
 * \param[in] mask The mask to apply to 'input'
 * \param[out] output The output buffer of the mask application
 * \param[in] size The number of pixels in one frame of 'input'
 * \param[in] batch_size The number of frames of 'input'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const cuComplex* __restrict__ input,
                const cuComplex* __restrict__ mask,
                cuComplex* __restrict__ output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* __restrict__ input,
                const float* __restrict__ mask,
                cuComplex* __restrict__ output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const float* __restrict__ input,
                const float* __restrict__ mask,
                float* __restrict__ output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);
