#ifndef FFT2_CUH
# define FFT2_CUH

#include <cufft.h>
#include "queue.hh"

void fft2_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z);
void fft_2(
  unsigned short* result_buffer,
  holovibes::Queue& q,
  cufftComplex *lens,
  float *sqrt_vect,
  cufftHandle plan3d,
  cufftHandle plan2d,
  unsigned int nbimages,
  unsigned int p);

#endif