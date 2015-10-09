#ifndef TOOLS_MULTIPLY_CUH
# define TOOLS_MULTIPLY_CUH

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

# include "geometry.hh"

/*! \brief  Multiply the pixels value of 2 complexe input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_complex(
  const cufftComplex* input1,
  const cufftComplex* input2,
  cufftComplex* output,
  unsigned int size);

/*! \brief  Multiply the pixels value of 2 float input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_float(
  const float* input1,
  const float* input2,
  float* output,
  unsigned int size);

#endif /* TOOLS_MULTIPLY.CUH */