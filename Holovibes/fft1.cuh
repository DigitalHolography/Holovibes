#ifndef FFT1_CUH
# define FFT1_CUH

#include <cufft.h>
#include "queue.hh"
#include "frame_desc.hh"

/*! \param lens Lens is externally allocated. */
void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z);
void fft_1(
  unsigned short *result_buffer,
  holovibes::Queue& q,
  cufftComplex *lens,
  float *sqrt_vect,
  cufftHandle plan,
  int nbimages);

#endif /* !FFT1_CUH */