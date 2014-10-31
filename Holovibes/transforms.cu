#include "transforms.cuh"

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