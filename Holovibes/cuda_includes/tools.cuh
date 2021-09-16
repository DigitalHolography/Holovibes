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

/*! \brief Circularly shifts the elements in input given a point(i,j) and the size of the frame. */
__global__ void circ_shift(const cuComplex* input,
                           cuComplex* output,
                           const uint batch_size,
                           const int i, // shift on x axis
                           const int j, // shift on y axis
                           const uint width,
                           const uint height,
                           const uint size);

/*! \brief Circularly shifts the elements in input given a point(i,j) given float output & inputs. */
__global__ void circ_shift_float(const float* input,
                                 float* output,
                                 const uint batch_size,
                                 const int i, // shift on x axis
                                 const int j, // shift on y axis
                                 const uint width,
                                 const uint height,
                                 const uint size);

__global__ void kernel_complex_to_modulus(const cuComplex* input,
                                          float* output,
                                          const uint size);