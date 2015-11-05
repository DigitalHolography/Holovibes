#include "tools_conversion.cuh"

#include <device_launch_parameters.h>
#include "hardware_limits.hh"

__global__ void img8_to_complex(
  cufftComplex* output,
  const unsigned char* input,
  const unsigned int size,
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
  const unsigned short* input,
  const unsigned int size,
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

/*! \brief Kernel function wrapped in complex_to_modulus, making
 ** the call easier
 **/
static __global__ void kernel_complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = hypotf(input[index].x, input[index].y);

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int threads = 128;
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_modulus << <blocks, threads >> >(input, output, size);
}

/*! \brief Kernel function wrapped in complex_to_squared_modulus, making
 ** the call easier
 **/
static __global__ void kernel_complex_to_squared_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input[index].x * input[index].x + input[index].y * input[index].y;

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_squared_modulus(
  const  cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_squared_modulus << <blocks, threads >> >(input, output, size);
}

/*! \brief Kernel function wrapped in complex_to_argument, making
 ** the call easier
 **/
static __global__ void kernel_complex_to_argument(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
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
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_argument << <blocks, threads >> >(input, output, size);
}

/*! \brief Kernel function wrapped in endianness_conversion, making
 ** the call easier
 **/
static __global__ void kernel_endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = (input[index] << 8) | (input[index] >> 8);

    index += blockDim.x * gridDim.x;
  }
}

void endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks - 1;

  kernel_endianness_conversion << <blocks, threads >> >(input, output, size);
}

/*! \brief Kernel function wrapped in float_to_ushort, making
 ** the call easier
 **/
static __global__ void kernel_float_to_ushort(
  const float* input,
  unsigned short* output,
  const unsigned int size)
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
  const float* input,
  unsigned short* output,
  const unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_float_to_ushort << <blocks, threads >> >(input, output, size);
}