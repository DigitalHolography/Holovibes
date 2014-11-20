#include "fft2.cuh"

#include <cuda_runtime.h>

#include "hardware_limits.hh"
#include "transforms.cuh"
#include "preprocessing.cuh"
#include "tools.cuh"

void fft2_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  float lambda,
  float z)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);

  kernel_spectral_lens<<<lblocks, lthreads>>>(lens, fd, lambda, z);
}

void fft_2(
  cufftComplex* input,
  cufftComplex* output,
  cufftComplex *lens,
  cufftHandle plan3d,
  cufftHandle plan2d,
  unsigned int frame_resolution,
  unsigned int nframes,
  unsigned int p)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (n_frame_resolution + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  cufftExecC2C(plan3d, input, input, CUFFT_FORWARD);

  cufftComplex* pframe = input + frame_resolution * p;

  kernel_apply_lens<<<blocks, threads>>>(
    pframe,
    frame_resolution,
    lens,
    frame_resolution);

  cufftExecC2C(plan2d, pframe, output, CUFFT_INVERSE);

  kernel_divide<<<blocks, threads >>>(output, frame_resolution, n_frame_resolution);
}
