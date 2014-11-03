#ifndef FFT2_CUH
#define FFT2_CUH

#include "string"
#include "tools.cuh"
#include "hardware_limits.hh"
#include "preprocessing.cuh"
#include "transforms.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

cufftComplex *create_spectral(float lambda, float distance, int size_x, int size_y, float pasu, float pasv);
void fft_2(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan);

#endif