#ifndef FFT1_CUH
# define FFT1_CUH

#include <cufft.h>
#include "queue.hh"
#include <frame_desc.hh>

/*! \param lens Lens is externally allocated. */
void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z);
void fft_1(
  cufftComplex* input,
  cufftComplex* lens,
  cufftHandle plan,
  unsigned int frame_resolution,
  unsigned int nframes);

#endif /* !FFT1_CUH */