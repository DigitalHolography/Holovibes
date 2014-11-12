#ifndef FFT2_CUH
# define FFT2_CUH

#include <cufft.h>
#include "queue.hh"

cufftComplex *create_spectral(
  float lambda,
  float distance,
  int size_x,
  int size_y,
  const camera::FrameDescriptor& fd);
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