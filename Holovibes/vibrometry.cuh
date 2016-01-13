/*! \file
 *
 * Hologram division (vibrometry) function. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

/*! \brief For each pixel (P and Q) of the two images, this function
* will output on output (O) : \n
* Ox = (PxQx + PyQy) / (QxQx + QyQy) \n
* Oy = (PyQx - PxQy) / (QxQx + QyQy) \n
*
* \param frame_p the numerator image
* \param frame_q the denominator image
*/
void frame_ratio(
  const cufftComplex* frame_p,
  const cufftComplex* frame_q,
  cufftComplex* output,
  const unsigned int size,
  cudaStream_t stream = 0);