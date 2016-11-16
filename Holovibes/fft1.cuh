/*! \file
 *
 * Functions that run the fft1 and the computation of the fft1 lens.
 */
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

/* Forward declaration. */
namespace camera
{
  struct FrameDescriptor;
}

/*! \brief Find the right threads and block to call quadratic lens
* with and call it
*/
void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  const float lambda,
  const float z,
  cudaStream_t stream = 0);

/*! \brief Apply a lens and call an fft1 on the image
*
* \param lens the lens that will be applied to the image
* \param plan the first paramater of cufftExecC2C that will be called
* on the image
*/
void fft_1(
  cufftComplex* input,
  const cufftComplex* lens,
  const cufftHandle plan1D,
  const cufftHandle plan2D,
  const unsigned int frame_resolution,
  const unsigned int nframes,
  const unsigned int p,
  const unsigned int q,
  cudaStream_t stream = 0);
