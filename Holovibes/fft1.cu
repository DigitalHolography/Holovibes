#include "fft1.cuh"

#include <cuda_runtime.h>
#include "hardware_limits.hh"
#include "tools.cuh"
#include "preprocessing.cuh"
#include "transforms.cuh"

void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);

  kernel_quadratic_lens <<<lblocks, lthreads>>>(lens, fd, lambda, z);
}

void fft_1(
  cufftComplex* input_buffer,
  unsigned short *result_buffer,
  holovibes::Queue& q,
  cufftComplex *lens,
  cufftHandle plan,
  unsigned int nbimages)
{
  // Sizes
  unsigned int pixel_size = q.get_frame_desc().width * q.get_frame_desc().height * nbimages;

  // Loaded images --> complex
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (pixel_size + threads - 1) / threads;

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  // Apply lens
  apply_quadratic_lens <<<blocks, threads>>>(input_buffer, pixel_size, lens, q.get_pixels());

  // FFT
  cufftExecC2C(plan, input_buffer, input_buffer, CUFFT_FORWARD);

  // Complex --> real (unsigned short)
  complex_2_module <<<blocks, threads>>>(input_buffer, result_buffer, pixel_size);
}