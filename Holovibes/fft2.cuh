/*! \file */
#ifndef FFT2_CUH
# define FFT2_CUH

#include <cufft.h>
#include "queue.hh"

/*! \brief Find the right threads and block to call spectral lens
* with and call it
*/
void fft2_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z);

/*! \brief takes input complex buffer and computes a p frame that is stored
 * at output pointer. The output pointer can be another complex buffer or the
 * same as input buffer.
 */
void fft_2(
  cufftComplex* input,
  cufftComplex* lens,
  cufftHandle plan3d,
  cufftHandle plan2d,
  unsigned int frame_resolution,
  unsigned int nframes,
  unsigned int p,
  unsigned int q);

#endif