#include "transforms.cuh"

__global__ void kernel_quadratic_lens(cufftComplex* output,
  unsigned int matrix_size,
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
  // FIXME pass sizex sizy as parameters
  float x = (i - (sqrtf(matrix_size) / 2)) * dx;
  float y = (j - (sqrtf(matrix_size) / 2)) * dy;

  if (index < matrix_size)
  {
    csquare = c * (x * x + y * y);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}