#ifndef TEST_CUH
#define TEST_CUH

#include "string"
#include "fourier_computing.cuh"
#include "tools.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


float *test_16(int nbimages, holovibes::Queue *q);
void img2disk(std::string path, void* img, unsigned int size);
void fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer);
cufftComplex* create_lens(unsigned int size_x, unsigned int size_y, float lambda, float z);
#endif