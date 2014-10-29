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
    output[index] = sqrtf(input[index].x * input[index].x + input[index].y * input[index].y);
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

__global__ void shift_corners(unsigned short *input, unsigned short *output, int size_x, int size_y)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size_x * size_y)
  {
    int x = index % size_x;
    int y = index / size_y;
    int n_x;
    int n_y;
    if (x < size_x / 2) // zone 1/3
    {
      if (y < size_y / 2) //zone 1
      {
        n_x = x + size_x / 2;
        n_y = y + size_y / 2;
      }
      else // zone 3
      {
        n_x = x + size_x / 2;
        n_y = y - size_y / 2;
      }
    }
    else // zone 2/4
    {
      if (y < size_y / 2) //zone 2
      {
        n_x = x - size_x / 2;
        n_y = y + size_y / 2;
      }
      else // zone 4
      {
        n_x = x - size_x / 2;
        n_y = y - size_y / 2;
      }
    }
    output[n_y * size_x + n_x] = input[index];
    index += blockDim.x * gridDim.x;
  }
}

void shift_corners(unsigned short **input, int size_x, int size_y)
{
  unsigned short *output;
  unsigned int size = size_x * size_y * sizeof(unsigned short);
  cudaMalloc(&output, size);
  int threads = get_max_threads_1d();
  int blocks = ((size_x * size_x) + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks();
  shift_corners << <blocks, threads >> >(*input, output, size_x, size_y);
  cudaMemcpy(*input, output, size, cudaMemcpyDeviceToDevice);
  cudaFree(output);
}

void complex_2_modul_call(cufftComplex* input, unsigned short* output, int size, int blocks, int threads)
{
  if (blocks > 65536)
  {
    blocks = 65536;
  }

  complex_2_module <<<blocks, threads >> >(input, output, size);
}
