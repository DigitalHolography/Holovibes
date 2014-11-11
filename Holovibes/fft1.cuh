#ifndef FFT1_CUH
# define FFT1_CUH

#include "tools.cuh"
#include "hardware_limits.hh"
#include "preprocessing.cuh"
#include "transforms.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan);
cufftComplex* create_lens(camera::FrameDescriptor fd, float lambda, float z);

#endif /* !FFT1_CUH */