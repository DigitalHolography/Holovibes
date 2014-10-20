#include "stdafx.h"
#include "tools.cuh"


// CONVERSION FUNCTIONS

__global__ void image_2_float(cufftReal* res, unsigned char* data, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    res[index] = (float)data[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void image_2_complex(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    res[index].x = data[index];//sqrt_tab[data[index]];
    res[index].y = data[index];//sqrt_tab[data[index]];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void complex_2_module(cufftComplex* input, unsigned char* output, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = sqrtf(input[index].x * input[index].x + input[index].y * input[index].y);
    index += blockDim.x * gridDim.x;
  }
}