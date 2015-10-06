#ifndef STFT_CUH
# define STFT_CUH

#include <cufft.h>
#include "queue.hh"

/*! Function handling the stft algorithm which steps are
 ** 1 : Do a fft2d (fresnel transform) on the input queue
 ** 2 : Take the ROI (i.e. 512x512px) and store it on complex queue
 ** 3 : Do the fft1 on images
 ** TODO : Implement it, because for the moment it is empty...
 **/
void stft(
  cufftComplex* input,
  cufftComplex* lens,
  cufftComplex* stft_buf,
  cufftHandle   plan2d,
  unsigned int frame_resolution,
  unsigned int nframes);

#endif