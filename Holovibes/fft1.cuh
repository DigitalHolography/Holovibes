/*! \file */
#ifndef FFT1_CUH
# define FFT1_CUH

#include <cufft.h>
#include "queue.hh"
#include <frame_desc.hh>

/*! \brief Find the right threads and block to call quadratic lens
* with and call it
*/
void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z);

/*! \brief Apply a lens and call an fft1 on the image
*
* \param lens the lens that will be applied to the image
* \param plan the first paramater of cufftExecC2C that will be called
* on the image
*/
void fft_1(
  cufftComplex* input,
  cufftComplex* lens,
  cufftHandle plan,
  unsigned int frame_resolution,
  unsigned int nframes);

#endif /* !FFT1_CUH */