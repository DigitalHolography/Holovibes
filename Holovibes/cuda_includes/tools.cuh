/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "tools.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"

/*! \brief Extract a part of the input image to the output.
 *
 * The memcpy is async
 *
 * \param input The full input image
 * \param zone the part of the image we want to extract
 * \param In pixels, the original width of the image
 * \param Where to store the cropped image
 * \param stream The CUDA stream on which to launch the operation.
 */
void frame_memcpy(const float* input,
                  const holovibes::units::RectFd& zone,
                  const uint input_width,
                  float* output,
                  const cudaStream_t stream);

__global__ void kernel_complex_to_modulus(const cuComplex* input, float* output, const uint size);

/*! \brief Circularly shifts the elements in the given input to point (shift_x, shift_y).
 *
 *  \param[out] output The buffer to store the output image.
 *  \param[in] input The input image.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] shift_x The x point to shift.
 *  \param[in] shift_y The y point to shift.
 *  \param[in] stream The CUDA stream to perform computations.
 */
void circ_shift(float* output, float* input, uint width, uint height, int shift_x, int shift_y, cudaStream_t stream);