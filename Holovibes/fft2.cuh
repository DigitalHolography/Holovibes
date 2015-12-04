/*! \file */
#pragma once

#include <cufft.h>

#include "queue.hh"

/*! \brief Find the right threads and block to call spectral lens
* with and call it
*/
void fft2_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  const float lambda,
  const float z,
  cudaStream_t stream = 0);

/*! \brief takes input complex buffer and computes a p frame that is stored
 * at output pointer. The output pointer can be another complex buffer or the
 * same as input buffer.
 */
void fft_2(
  cufftComplex* input,
  const cufftComplex* lens,
  const cufftHandle plan3d,
  const cufftHandle plan2d,
  const unsigned int frame_resolution,
  const unsigned int nframes,
  const unsigned int p,
  const unsigned int q,
  cudaStream_t stream = 0);