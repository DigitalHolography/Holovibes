/*! \file */
#ifndef TRANSFORMS_CUH_
# define TRANSFORMS_CUH_

# include <cufft.h>
# include <frame_desc.hh>

/*! \brief Compute a lens to apply to an image used by the fft1
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on which the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_quadratic_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float dist);

/*! \brief Compute a lens to apply to an image used by the fft2
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on wich the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_spectral_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float distance);

#endif /* !TRANSFORMS_CUH_ */