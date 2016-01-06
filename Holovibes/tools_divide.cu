#include <device_launch_parameters.h>

#include "tools_divide.cuh"
#include "hardware_limits.hh"

__global__ void kernel_complex_divide(
  cufftComplex* image,
  const unsigned int size,
  const float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_float_divide(
  float* input,
  const unsigned int size,
  const float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    input[index] /= divider;
    index += blockDim.x * gridDim.x;
  }
}