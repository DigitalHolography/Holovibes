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
void complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size);
void complex_to_squared_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size);
void complex_to_argument(
  cufftComplex* input,
  float* output,
  unsigned int size);
__global__ void kernel_apply_lens(
  cufftComplex *input,
  unsigned int input_size,
  cufftComplex *lens,
  unsigned int lens_size);
// TODO: Explain what this does.
__global__ void kernel_divide(
  cufftComplex* image,
  unsigned int size,
  float divider);
void shift_corners(
  float *input,
  unsigned int size_x,
  unsigned int size_y);
void endianness_conversion(
  unsigned short* input,
  unsigned short* output,
  unsigned int size);
void apply_log10(
  float* input,
  unsigned int size);
void float_to_ushort(
  float* input,
  unsigned short* output,
  unsigned int size);

#endif /* !TOOLS_CUH */