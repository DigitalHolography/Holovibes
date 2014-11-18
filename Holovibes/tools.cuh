#ifndef TOOLS_CUH
# define TOOLS_CUH

# include <cuda_runtime.h>
# include <cufft.h>

#ifndef _USE_MATH_DEFINES
/* Enables math constants. */
# define _USE_MATH_DEFINES
#endif /* !_USE_MATH_DEFINES */
#include <math.h>

// CONVERSION FUNCTIONS
__global__ void img8_to_complex(
  cufftComplex* output,
  unsigned char* input,
  unsigned int size,
  const float* sqrt_array);
__global__ void img16_to_complex(
  cufftComplex* output,
  unsigned short* input,
  unsigned int size,
  const float* sqrt_array);
// TODO: Rename 'module' to 'modulus'.
__global__ void complex_2_module(cufftComplex* input, unsigned short* output, int size);
// TODO: Rename 'magnitude' to 'modulus'.
__global__ void complex_2_squared_magnitude(cufftComplex* input, unsigned short* output, int size);
__global__ void complex_2_argument(cufftComplex* input, unsigned short* output, int size);
__global__ void apply_quadratic_lens(cufftComplex *input, int input_size, cufftComplex *lens, int lens_size);
__global__ void divide(cufftComplex* image, int size_x, int size_y, int nbimages);
void shift_corners(unsigned short *input, int size_x, int size_y);
void endianness_conversion(unsigned short* input, unsigned short* output, unsigned int size);

#endif /* !TOOLS_CUH */