#ifndef TOOLS_CUH_
# define TOOLS_CUH_

# include <cuda_runtime.h>
# include <cufft.h>

#ifndef _USE_MATH_DEFINES
/* Enables math constants. */
# define _USE_MATH_DEFINES
#endif /* !_USE_MATH_DEFINES */
#include <math.h>

// CONVERSION FUNCTIONS
__global__ void image_2_complex8(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab);
__global__ void image_2_complex16(cufftComplex* res, unsigned short* data, int size, float *sqrt_tab);
__global__ void complex_2_module(cufftComplex* input, unsigned short* output, int size);
__global__ void apply_quadratic_lens(cufftComplex *input, int input_size, cufftComplex *lens, int lens_size);
__global__ void meshgrind_square(float *input_u, float *input_v, float *output_u, float *output_v, unsigned int size_x, unsigned int size_y);
__global__ void fft2_make_u_v(float pasu, float pasv, float *u, float *v, unsigned int size_x, unsigned int size_y);
__global__ void divide(cufftComplex* image, int size_x, int size_y, int nbimages);
void shift_corners(unsigned short **input, int size_x, int size_y);
void endianness_conversion(unsigned short* input, unsigned short* output, unsigned int size);

#endif /* !TOOLS_CUH_ */