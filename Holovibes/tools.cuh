#ifndef TOOLS_CUH_
# define TOOLS_CUH_

# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <cufft.h>
# include <cufftXt.h>
# include <cufftw.h>
# include "hardware_limits.hh"

// CONVERSION FUNCTIONS
__global__ void image_2_float(cufftReal* res, unsigned char* data, int size);
__global__ void image_2_float(cufftReal* res, unsigned short* data, int size);
__global__ void image_2_complex8(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab);
__global__ void image_2_complex16(cufftComplex* res, unsigned short* data, int size, float *sqrt_tab);
__global__ void complex_2_module(cufftComplex* input, unsigned short* output, int size);
__global__ void apply_quadratic_lens(cufftComplex *input, int input_size, cufftComplex *lens, int lens_size);
void complex_2_modul_call(cufftComplex* input, unsigned short* output, int size, int blocks, int threads);
void complex_2_modul_call(cufftComplex* input, unsigned char* output, int size, int blocks, int threads);
void shift_corners(unsigned short **input, int size_x, int size_y);

#endif /* !TOOLS_CUH_ */