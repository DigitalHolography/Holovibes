#ifndef TOOLS_CUH
# define TOOLS_CUH

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

# include "geometry.hh"

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
__global__ void kernel_complex_divide(
  cufftComplex* image,
  unsigned int size,
  float divider);
__global__ void kernel_float_divide(
  float* input,
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

__global__ void kernel_multiply_frames_complex(
  const cufftComplex* input1,
  const cufftComplex* input2,
  cufftComplex* output,
  unsigned int size);

__global__ void kernel_multiply_frames_float(
  const float* input1,
  const float* input2,
  float* output,
  unsigned int size);

/*! x, k, tmp_x, tmp_k and out shares the same resolution.
 * x, k remains unchanged.
 * \note This is a naive implementation of the convolution
 * operator. Some FFT could be avoided and reduce the memory
 * consumption, BUT implement a convolution operator (that let's
 * the code clear) is no longer possible since each optimization
 * is case dependant. */
void convolution_operator(
  const cufftComplex* x,
  const cufftComplex* k,
  float* out,
  unsigned int size,
  cufftHandle plan2d_x,
  cufftHandle plan2d_k);

/* Copy the content of input's zone to output frame.
 * It assumes the zone is contained inside the input frame
 * and output frame is large enough. */
void frame_memcpy(
  const float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width);

float average_operator(
  const float* input,
  const unsigned int size);

#endif /* !TOOLS_CUH */