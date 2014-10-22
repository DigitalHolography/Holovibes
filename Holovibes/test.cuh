#ifndef TEST_CUH
#define TEST_CUH

#include "string"
#include "preprocessing.cuh"
#include "fourier_computing.cuh"
#include "tools.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void test_fft(int nbimages, holovibes::Queue *q);
float *test_16(int nbimages, holovibes::Queue *q);
void img2disk(std::string path, void* img, unsigned int size);
#endif