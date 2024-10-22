/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief This function applies a mask to a number of frames
 *
 * \param input The buffer of images to modify
 * \param mask The mask to apply to 'input'
 * \param size The number of pixels in one frame of 'input'
 * \param batch_size The number of frames of 'input'
 * \param stream The CUDA stream on which to launch the operation.
 */
void apply_mask(
    cuComplex* in_out, const cuComplex* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

void apply_mask(
    cuComplex* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

void apply_mask(float* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param input The buffer of images to modify
 * \param mask The mask to apply to 'input' stored in 'output'
 * \param output The output buffer of the mask application
 * \param size The number of pixels in one frame of 'input'
 * \param batch_size The number of frames of 'input'
 * \param stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const cuComplex* input,
                const cuComplex* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const cuComplex* input,
                const float* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

void apply_mask(const float* input,
                const float* mask,
                float* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

/*! \brief Computes the mean of the pixels inside the image only if the pixel is in the given mask.
 *  Calls a CUDA Kernel.
 *
 *  \param[in] input The input image on which the mask is applied and the mean of pixels is computed.
 *  \param[in] mask The mean will be computed only inside this mask.
 *  \param[in out] pixels_mean Pointer to store the mean of the pixels inside the circle.
 *  \param[in] size The size of the image, e.g : width x height.
 *  \param[in] stream The CUDA stream on which to launch the operation.
 */
void get_mean_in_mask(
    const float* input, const float* mask, float* pixels_mean, const size_t size, const cudaStream_t stream);