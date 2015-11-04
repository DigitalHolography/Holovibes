/*! \file */
#ifndef TRANSFORMS_CUH_
# define TRANSFORMS_CUH_

# include <cufft.h>
# include <frame_desc.hh>

__global__ void kernel_quadratic_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float dist);
__global__ void kernel_spectral_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float distance);

#endif /* !TRANSFORMS_CUH_ */