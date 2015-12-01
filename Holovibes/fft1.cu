#include <cuda_runtime.h>

#include "fft1.cuh"
#include "hardware_limits.hh"
#include "tools.cuh"
#include "preprocessing.cuh"
#include "transforms.cuh"

void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  const float lambda,
  const float z)
{
  unsigned int threads = 128;
  unsigned int blocks = (fd.frame_res() + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();
  kernel_quadratic_lens << <blocks, threads >> >(lens, fd, lambda, z);
}

void fft_1(
  cufftComplex* input,
  const cufftComplex* lens,
  const cufftHandle plan,
  const unsigned int frame_resolution,
  const unsigned int nframes)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = n_frame_resolution / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  // Apply lens on multiple frames.
  kernel_apply_lens << <blocks, threads >> >(input, n_frame_resolution, lens, frame_resolution);
  // FFT
  cufftExecC2C(plan, input, input, CUFFT_FORWARD);

  cudaDeviceSynchronize();
}