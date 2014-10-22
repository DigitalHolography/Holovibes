#include "transforms.cuh"

__global__ void kernel_quadratic_lens(cufftComplex* output,
  unsigned int xsize,
  unsigned int ysize,
  float lambda,
  float dist)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  float c = M_PI / (lambda * dist);
  float csquare;
  float dx = 5.5e-6f;
  float dy = 5.5e-6f;

  float x = (i - ((float)xsize / 2)) * dx;
  float y = (j - ((float)ysize / 2)) * dy;

  if (index < xsize * ysize)
  {
    csquare = c * (x * x + y * y);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}