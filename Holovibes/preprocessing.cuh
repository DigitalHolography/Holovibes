/*! \file */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

namespace holovibes
{
  class Queue;
}

/*! \brief Precompute the sqrt q sqrt vector of values in
* range 0 to n.
*
* \param n Number of values to compute.
* \param output Array of the sqrt values form 0 to n - 1,
* this array should have size greater or equal to n.
*/
void make_sqrt_vect(float* out,
  const unsigned short n,
  cudaStream_t stream = 0);

/*! \brief Ensure the contiguity of images extracted from
 * the queue for any further processing.
 * This function also compute the sqrt value of each pixel of images.
 *
 * \param input the device queue from where images should be taken
 * to be processed.
 * \param output A bloc made of n contigus images requested
 * to the function.
 * \param n Number of images to ensure contiguity.
 * \param sqrt_array Array of the sqrt values form 0 to 65535
 * in case of 16 bit images or from 0 to 255 in case of
 * 8 bit images.
 *
 *
 *\note This function can be improved by specifying
 * img8_to_complex or img16_to_complex in the pipe to avoid
 * branch conditions. But it is no big deal.
 * Otherwise, the convert function are not called outside because
 * this function would need an unsigned short buffer that is unused
 * anywhere else.
 */
void make_contiguous_complex(
  holovibes::Queue& input,
  cufftComplex* output,
  const unsigned int n,
  const float* sqrt_array,
  cudaStream_t stream = 0);