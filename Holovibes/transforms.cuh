#ifndef TRANSFORMS_CUH_
# define TRANSFORMS_CUH_

# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <cufft.h>
# include <cufftXt.h>
# include <cufftw.h>
/* Mandatory to use math.h constants (such as pi) */
#define _USE_MATH_DEFINES
# include <math.h>
# include "camera.hh"

__global__ void kernel_quadratic_lens(cufftComplex* output,
  camera::FrameDescriptor fd,
  float lambda,
  float dist);

#endif /* !TRANSFORMS_CUH_ */