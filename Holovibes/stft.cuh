#ifndef STFT_CUH
# define STFT_CUH

# include <cufft.h>
# include "queue.hh"
# include "geometry.hh"
# include "compute_descriptor.hh"

/*! Function handling the stft algorithm which steps are
 ** 1 : Aplly lens on the input queue
 ** 2 : Do a fft2d (fresnel transform) on the input queue
 ** 3 : Take the ROI (i.e. 512x512px) and store bursting way on a complex queue (stft_buf)
 ** 4 : Do nsamples fft1d on the complex queue (stft_dup_buf)
 ** This complex queue need to be reconstruct in order to get image
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

/*! Reconstruct image from bursting complex queue (stft_dup_buf)
 ** And rescale it to reconstruct width/height
 **/

void stft_recontruct(
  cufftComplex*                   output,
  cufftComplex*                   stft_dup_buf,
  const holovibes::Rectangle&     r,
  const camera::FrameDescriptor&  desc,
  unsigned int                    reconstruct_width,
  unsigned int                    reconstruct_height,
  unsigned int                    pindex,
  unsigned int                    nsamples);
#endif