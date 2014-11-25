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

static __global__ void kernel_complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = sqrtf(input[index].x * input[index].x + input[index].y * input[index].y);

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_modulus<<<blocks, threads>>>(input, output, size);
}

static __global__ void kernel_complex_to_squared_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input[index].x * input[index].x + input[index].y * input[index].y;

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_squared_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_squared_modulus<<<blocks, threads>>>(input, output, size);
}

static __global__ void kernel_complex_to_argument(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float pi_div_2 = M_PI / 2.0f;
  float c = 65535.0f / M_PI;

  while (index < size)
  {
    output[index] = (atanf(input[index].y / input[index].x) + pi_div_2) * c;

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_argument(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_argument<<<blocks, threads>>>(input, output, size);
}

__global__ void kernel_apply_lens(
  cufftComplex *input,
  unsigned int input_size,
  cufftComplex *lens,
  unsigned int lens_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    input[index].x = input[index].x * lens[index % lens_size].x;
    input[index].y = input[index].y * lens[index % lens_size].y;
    index += blockDim.x * gridDim.x;
  }
}

static __global__ void kernel_shift_corners(
  unsigned short* input,
  unsigned int size_x,
  unsigned int size_y)
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

void shift_corners(
  unsigned short* input,
  unsigned int size_x,
  unsigned int size_y)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

  kernel_shift_corners <<< lblocks, lthreads >>>(input, size_x, size_y);
}

static __global__ void kernel_endianness_conversion(
  unsigned short* input,
  unsigned short* output,
  size_t size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = (input[index] << 8) | (input[index] >> 8);

    index += blockDim.x * gridDim.x;
  }
}

void endianness_conversion(
  unsigned short* input,
  unsigned short* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks - 1;

  kernel_endianness_conversion <<<blocks, threads >>>(input, output, size);
}

__global__ void kernel_divide(
  cufftComplex* image,
  unsigned int size,
  float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_log10(
  float* input,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  const float scale = 65535.0f / log10f(65535.0f);

  while (index < size)
  {
    input[index] = log10f(input[index]) * scale;

    index += blockDim.x * gridDim.x;
  }
}

void apply_log10(
  float* input,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_log10<<<blocks, threads>>>(input, size);
}

static __global__ void kernel_float_to_ushort(
  float* input,
  unsigned short* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    if (input[index] > 65535.0f)
      output[index] = 65535;
    else if (input[index] < 0.0f)
      output[index] = 0;
    else
      output[index] = static_cast<unsigned short>(input[index]);

    index += blockDim.x * gridDim.x;
  }
}

void float_to_ushort(
  float* input,
  unsigned short* output,
  unsigned int size)
{

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_float_to_ushort<<<blocks, threads>>>(input, output, size);
}