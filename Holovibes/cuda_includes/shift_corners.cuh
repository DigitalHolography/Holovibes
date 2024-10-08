/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "cuda_runtime.h"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to modify in-place.
 * \param batch_size Number of images in input
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(
    float* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream);

void shift_corners(
    cuComplex* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream);

void shift_corners(
    float3* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream);

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to shift.
 * \param output The destination image
 * \param batch_size Number of images in input
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(const float3* input,
                   float3* output,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream);

void shift_corners(const float* input,
                   float* output,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream);

void shift_corners(const cuComplex* input,
                   cuComplex* output,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream);
