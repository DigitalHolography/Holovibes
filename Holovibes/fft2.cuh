#ifndef FFT2_CUH
#define FFT2_CUH

#include "string"
#include "tools.cuh"
#include "hardware_limits.hh"
#include "preprocessing.cuh"
#include "contrast_correction.cuh"
#include "transforms.cuh"
#include "fft1.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



cufftComplex *create_spectral(float lambda, float distance, int size_x, int size_y, float pasx, float pasy, camera::FrameDescriptor fd);
void fft_2(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan3d, unsigned int p, cufftHandle plan2d);

#endif