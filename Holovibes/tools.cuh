#ifndef TOOLS_CUH_
# define TOOLS_CUH_

# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include <cufft.h>
# include <cufftXt.h>
# include <cufftw.h>

// CONVERSION FUNCTIONS
__global__ void image_2_float(cufftReal* res, unsigned char* data, int size);
__global__ void image_2_float(cufftReal* res, unsigned short* data, int size);
__global__ void image_2_complex(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab);
__global__ void image_2_complex(cufftComplex* res, unsigned short* data, int size, float *sqrt_tab);
__global__ void complex_2_module(cufftComplex* input, unsigned char* output, int size);
__global__ void complex_2_module(cufftComplex* input, unsigned short* output, int size);
__global__ void apply_quadratic_lens(cufftComplex *input, int input_size, cufftComplex *lens, int lens_size);

#endif /* !TOOLS_CUH_ */