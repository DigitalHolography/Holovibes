#include "transforms.cuh"

#include <device_launch_parameters.h>

#ifndef _USE_MATH_DEFINES
/* Enables math constants. */
# define _USE_MATH_DEFINES
#endif /* !_USE_MATH_DEFINES */
#include <math.h>

__global__ void kernel_quadratic_lens(cufftComplex* output,
  camera::FrameDescriptor fd,
  float lambda,
  float dist)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  float c = M_PI / (lambda * dist);
  float csquare;
  float dx = fd.pixel_size * 1.0e-6f;
  float dy = fd.pixel_size * 1.0e-6f;

  float x = (i - ((float)fd.width / 2)) * dx;
  float y = (j - ((float)fd.height / 2)) * dy;

  if (index < fd.width * fd.height)
  {
    csquare = c * (x * x + y * y);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}

__global__ void kernel_spectral_lens(cufftComplex* output,
  camera::FrameDescriptor fd,
  float lambda,
  float distance)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  float c = 2 * M_PI * distance / lambda;
  float csquare;

  float dx = fd.pixel_size * 1.0e-6f;
  float dy = fd.pixel_size * 1.0e-6f;

  float du = 1 / (((float)fd.width) * dx);
  float dv = 1 / (((float)fd.height) * dy);

  float u = (i - ((float)fd.width / 2)) * du; //fix me -1
  float v = (j - ((float)fd.height / 2)) * dv; //fix me -1

  if (index < fd.width * fd.height)
  {
    csquare = c * sqrtf(1.0f - lambda * lambda * u * u - lambda * lambda * v * v);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}

__global__ void spectral(float pasu, float pasv, cufftComplex *output, unsigned int size_x, unsigned int size_y, float lambda, float distance)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float rounded = 0;
  float to_round = index / 2;
  int entire_part = floor(to_round);
  float decimal_part = to_round - entire_part;

  if (decimal_part >= 0.5)
    rounded = entire_part + 1;
  else
    rounded = entire_part;

  int x = index % size_x;
  int y = index / size_y;
  float u = ((x - 1) - rounded) * pasu;
  float v = ((y - 1) - rounded) * pasv;

  if (index < size_x * size_y)
  {
    float u_square = u * u;
    float v_square = v * v;
    float lambda_square = lambda * lambda;
    float thetha = 2 * M_PI * distance / lambda * sqrt(1 - (lambda_square * u_square) - (lambda_square * v_square));
    output[index].x = cosf(thetha);
    output[index].y = sinf(thetha);
  }
}

__global__ void spectral_rework1(cufftComplex *input, cufftComplex *output, int size_x, int size_y)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size_x * size_y)
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
        output[n_y * size_x + n_x] = input[index];
      }
    }
  }
}

__global__ void spectral_rework2(cufftComplex *input, cufftComplex *output, int size_x, int size_y)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size_x * size_y)
  {
    int x = index % size_x;
    int y = index / size_y;
    int n_x;
    int n_y;
    if (x > size_x / 2) // zone 1/3
    {
      if (y < size_y / 2) //zone 1
      {
        n_x = x + size_x / 2;
        n_y = y + size_y / 2;
        output[n_y * size_x + n_x] = input[index];
      }
    }
  }
}
