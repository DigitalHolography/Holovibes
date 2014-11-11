#include "stdafx.h"
#include "tools.cuh"

// CONVERSION FUNCTIONS

__global__ void image_2_complex8(cufftComplex* res, unsigned char* data, int size, float *sqrt_tab)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    // Image rescaling on 2^16 colors (65535 / 255 = 257)
    unsigned int val = sqrt_tab[data[index] * 257];
    res[index].x = val;
    res[index].y = val;
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
    float m = sqrtf(input[index].x * input[index].x + input[index].y * input[index].y);

    if (m > 65535.0f)
      output[index] = 65535;
    else
      output[index] = m;

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
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = ((size_x * size_x) + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks();
  shift_corners << <blocks, threads >> >(*input, output, size_x, size_y);
  cudaMemcpy(*input, output, size, cudaMemcpyDeviceToDevice);
  cudaFree(output);
}

__global__ void kernel_endianness_conversion(unsigned short* input, unsigned short* output, size_t size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = (input[index] << 8) | (input[index] >> 8);

    index += blockDim.x * gridDim.x;
  }
}

__global__ void fft2_make_u_v(float pasu, float pasv, float *u, float *v, unsigned int size_x, unsigned int size_y)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size_x || index < size_y)
  {
    float rounded = 0;
    float to_round = index / 2;
    int entire_part = floor(to_round);
    float decimal_part = to_round - entire_part;
    if (decimal_part >= 0.5)
      rounded = entire_part + 1;
    else
      rounded = entire_part;

    if (index < size_x)
      u[index] = ((index - 1) - rounded) * pasu;
    if (index < size_y)
      v[index] = ((index - 1) - rounded) * pasv;
    index += blockDim.x * gridDim.x;
  }
}

// output_u size_x * size_y
// output_v size_x * size_y
__global__ void meshgrind_square(float *input_u, float *input_v, float *output_u, float *output_v, unsigned int size_x, unsigned int size_y)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size_x * size_y)
  {
    output_u[index] = input_u[index % size_x] * input_u[index % size_x];
    output_v[index] = input_v[index / size_y] * input_v[index / size_y];
    index += blockDim.x * gridDim.x;
  }
}

__global__ void divide(cufftComplex* image, int size_x, int size_y, int nbimages)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size_x * size_y)
  {
    image[index].x = image[index].x / ((float)nbimages * (float)size_x * (float)size_y);
    image[index].y = image[index].y / ((float)nbimages * (float)size_x * (float)size_y);
    index += blockDim.x * gridDim.x;
  }
}

void endianness_conversion(unsigned short* input, unsigned short* output, unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks - 1;

  kernel_endianness_conversion <<<blocks, threads >>>(input, output, size);
}