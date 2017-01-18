#include "transforms.cuh"
#include "frame_desc.hh"

__global__ void kernel_quadratic_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  const float lambda,
  const float dist)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int size = fd.width * fd.height;

  const float c = M_PI / (lambda * dist);

  const float dx = fd.pixel_size * 1.0e-6f;
  const float dy = fd.pixel_size * 1.0e-6f;

  float	x;
  float	y;
  unsigned int i;
  unsigned int j;
  float csquare;

  while (index < size)
  {
    i = index % fd.width;
    j = index / fd.height;
    x = (i - (static_cast<float>(fd.width >> 1))) * dx;
    y = (j - (static_cast<float>(fd.height >> 1))) * dy;

    csquare = c * (x * x + y * y);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_spectral_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  const float lambda,
  const float distance)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  const float c = M_2PI * distance / lambda;

  const float dx = fd.pixel_size * 1.0e-6f;
  const float dy = fd.pixel_size * 1.0e-6f;

  const float du = 1 / ((static_cast<float>(fd.width)) * dx);
  const float dv = 1 / ((static_cast<float>(fd.height)) * dy);

  const float u = (i - static_cast<float>(lrintf(static_cast<float>(fd.width >> 1)))) * du;
  const float v = (j - static_cast<float>(lrintf(static_cast<float>(fd.height >> 1)))) * dv;

  float csquare;
  if (index < fd.width * fd.height)
  {
    csquare = c * sqrtf(abs(1.0f - lambda * lambda * u * u - lambda * lambda * v * v));
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}