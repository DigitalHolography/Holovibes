#ifndef TOOLS_DIVIDE_CUH
# define TOOLS_DIVIDE_CUH

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

# include "geometry.hh"

/*! \brief  Divide all the pixels of input image(s) in complex representation by the float divider.
*
* The image(s) to treat, seen as image, should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* NB: doesn't work on architechture 2.5 in debug mod on GTX 470 graphic card
*/
__global__ void kernel_complex_divide(
  cufftComplex* image,
  unsigned int size,
  float divider);

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* NB: doesn't work on architechture 2.5 in debug mod on GTX 470 graphic card
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
__global__ void kernel_float_divide(
  float* input,
  unsigned int size,
  float divider);

#endif /* !TOOLS_DIVIDE_CUH */