#include "stdafx.h"
#include "tools.cuh"

#include <device_launch_parameters.h>
#include "hardware_limits.hh"

// CONVERSION FUNCTIONS

__global__ void img8_to_complex(
  cufftComplex* output,
  unsigned char* input,
  unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    // Image rescaling on 2^16 colors (65535 / 255 = 257)
    unsigned int val = sqrt_array[input[index] * 257];
    output[index].x = val;
    output[index].y = val;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void img16_to_complex(
  cufftComplex* output,
  unsigned short* input,
  unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index].x = sqrt_array[input[index]];
    output[index].y = sqrt_array[input[index]];
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

__global__ void complex_2_squared_magnitude(cufftComplex* input, unsigned short* output, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    float m = input[index].x * input[index].x + input[index].y * input[index].y;

    if (m > 65535.0f)
      output[index] = 65535;
    else
      output[index] = m;

    index += blockDim.x * gridDim.x;
  }
}

__global__ void complex_2_argument(cufftComplex* input, unsigned short* output, int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float pi_div_2 = M_PI / 2.0f;
  float c = 65535.0f / M_PI;

  while (index < size)
  {
    float m = (atanf(input[index].y / input[index].x) + pi_div_2) * c;

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

__global__ void kernel_shift_corners(unsigned short* input, unsigned int size_x, unsigned int size_y)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;
  unsigned int ni = 0;
  unsigned int nj = 0;
  unsigned int nindex = 0;

  // Superior half of the matrix
  if (j > size_y / 2)
  {
    // Left superior quarter of the matrix
    if (i < size_x / 2)
    {
      ni = i + size_x / 2;
      nj = j - size_y / 2;
    }
    // Right superior quarter
    else
    {
      ni = i - size_x / 2;
      nj = j - size_y / 2;
    }

    nindex = nj * size_x + ni;

    input[nindex] ^= input[index];
    input[index] ^= input[nindex];
    input[nindex] ^= input[index];
  }
}

void shift_corners(unsigned short* input, int size_x, int size_y)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

  kernel_shift_corners <<< lblocks, lthreads >>>(input, size_x, size_y);
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