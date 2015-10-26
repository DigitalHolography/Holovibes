#ifndef STFT_CUH
# define STFT_CUH

# include <cufft.h>
# include "queue.hh"
# include "geometry.hh"
# include "compute_descriptor.hh"

/*! Function handling the stft algorithm which steps are
 ** 1 : Do a fft2d (fresnel transform) on the input queue
 ** 2 : Take the ROI (i.e. 512x512px) and store it on complex queue
 ** 3 : Do the fft1 on images
 ** TODO : Implement it, because for the moment it is empty...
 **/

void stft(
  cufftComplex*                   input,
  cufftComplex*                   lens,
  cufftComplex*                   stft_buf,
  cufftComplex*                   stft_dup_buf,
  cufftHandle                     plan2d,
  cufftHandle                     plan1d,
  const holovibes::Rectangle&     r,
  unsigned int&                   curr_elt,
  const camera::FrameDescriptor&  desc,
  unsigned int                    nsamples);

void stft_recontruct(
  cufftComplex*                   input,
  cufftComplex*                   stft_dup_buf,
  const holovibes::Rectangle&     r,
  const camera::FrameDescriptor&  desc,
  unsigned int                    reconstruct_width,
  unsigned int                    reconstruct_height,
  unsigned int                    pindex,
  unsigned int                    nsamples);
#endif