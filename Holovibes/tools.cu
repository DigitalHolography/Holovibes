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

__global__ void image_2_float(cufftReal* res, unsigned short* data, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    res[index] = (float)data[index];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void image_2_complex8(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    res[index].x = sqrt_tab[data[index]];
    res[index].y = sqrt_tab[data[index]];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void image_2_complex16(cufftComplex* res, unsigned short* data, int size, float *sqrt_tab)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    res[index].x = sqrt_tab[data[index]];
    res[index].y = sqrt_tab[data[index]];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void complex_2_module(cufftComplex* input, unsigned short* output, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = log10(sqrtf(input[index].x * input[index].x + input[index].y * input[index].y)); //racine ? log
    index += blockDim.x * gridDim.x;
  }
}

__global__ void apply_quadratic_lens(cufftComplex *input, int input_size, cufftComplex *lens, int lens_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    input[index].x = input[index].x * lens[index % lens_size].x;
    input[index].y = input[index].y * lens[index % lens_size].y;
    index += blockDim.x * gridDim.x;
  }
}

void complex_2_modul_call(cufftComplex* input, unsigned short* output, int size, int blocks, int threads)
{
  if (blocks > 65536)
  {
    blocks = 65536;
  }

  complex_2_module <<<blocks, threads >> >(input, output, size);
}