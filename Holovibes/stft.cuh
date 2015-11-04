/*! \file */
#ifndef STFT_CUH
# define STFT_CUH

# include <cufft.h>
# include "queue.hh"
# include "geometry.hh"
# include "compute_descriptor.hh"

/*! \brief Function handling the stft algorithm which steps are \n
* 1 : Aplly lens on the input queue \n
* 2 : Do a fft2d (fresnel transform) on the input queue \n
* 3 : Take the ROI (i.e. 512x512px) and store bursting way on a complex queue (stft_buf) \n
* 4 : Do nsamples fft1d on the complex queue (stft_dup_buf) \n
* This complex queue need to be reconstruct in order to get image
*
* \param stft_buf the buffer which will be exploded
* \param stft_dup_buf the buffer that will receive the plan1d transforms
* \param r The rectangle selected for the stft to be done on it
*
* \verbatim

 [a1][b1]   [a2][b2]   [a3][b3]
 [c1][d1]   [c2][d2]   [c3][d3]
  img 1       img 2      img 3

       **   Bursting  **
 [a1][a2][a3][b1][b2][b3][c1][...]

       **   plan1d    **
    [a123][b123][c123][d123]

       ** Reconstruct **
         [a123][b123]
         [c123][d123]


\endverbatim
*/
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

/*! \brief Reconstruct image from bursting complex queue (stft_dup_buf)
* and rescale it to reconstruct width/height
*
* \param r the selected zone for stft
* \param pindex which image are we on
* \param nspamples how many images are used for transform
*/
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