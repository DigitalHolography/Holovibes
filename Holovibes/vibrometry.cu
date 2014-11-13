#include "vibrometry.cuh"
#include <device_launch_parameters.h>


__global__ void kernel_vibro(cufftComplex *image_p, cufftComplex *image_q, cufftComplex *output, unsigned int nb_pixels)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < nb_pixels)
  {
    output[index].x = image_p[index].x / image_q[index].x;
    output[index].y = image_p[index].y / image_q[index].y;
    index += blockDim.x * gridDim.x;
  }
}

cufftComplex *vibrometry(unsigned int p, unsigned int q, cufftComplex *images, const camera::FrameDescriptor fd)
{
  unsigned int pixels = fd.frame_res();
  unsigned int size = fd.frame_size();
  cufftComplex *output;
  cudaMalloc(&output, size);
  cufftComplex *image_p = images + pixels * p;
  cufftComplex *image_q = images + pixels * q;

  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks - 1;
  kernel_vibro<<<blocks,threads>>>(image_p, image_q, output, pixels);
  return output;
}

