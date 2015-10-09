#include "tools_multiply.cuh"

#include <device_launch_parameters.h>
#include "hardware_limits.hh"

__global__ void kernel_multiply_frames_complex(
  const cufftComplex* input1,
  const cufftComplex* input2,
  cufftComplex* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index].x = input1[index].x * input2[index].x;
    output[index].y = input1[index].y * input2[index].y;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_multiply_frames_float(
  const float* input1,
  const float* input2,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input1[index] * input2[index];
    index += blockDim.x * gridDim.x;
  }
}