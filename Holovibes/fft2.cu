#include "fft2.cuh"

#include <cuda_runtime.h>

#include "hardware_limits.hh"
#include "transforms.cuh"
#include "preprocessing.cuh"
#include "tools_divide.cuh"
#include "tools.cuh"

void fft2_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  const float lambda,
  const float z,
  cudaStream_t stream)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);

  kernel_spectral_lens << <lblocks, lthreads, 0, stream >> >(lens, fd, lambda, z);
}

void fft_2(
  cufftComplex* input,
  const cufftComplex* lens,
  const cufftHandle plan3d,
  const cufftHandle plan2d,
  const unsigned int frame_resolution,
  const unsigned int nframes,
  const unsigned int p,
  const unsigned int q,
  cudaStream_t stream)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = 128;
  unsigned int blocks = n_frame_resolution / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  cufftExecC2C(plan3d, input, input, CUFFT_FORWARD);

  cufftComplex* pframe = input + frame_resolution * p;
  cufftComplex* qframe = input + frame_resolution * q;

  cudaStreamSynchronize(stream);

  kernel_apply_lens << <blocks, threads, 0, stream >> >(
    input,
    n_frame_resolution,
    lens,
    frame_resolution);

  cudaStreamSynchronize(stream);

  cufftExecC2C(plan2d, pframe, pframe, CUFFT_INVERSE);
  kernel_complex_divide << <blocks, threads, 0, stream >> >(pframe, frame_resolution, static_cast<float>(n_frame_resolution));
  if (p != q)
  {
    cufftExecC2C(plan2d, qframe, qframe, CUFFT_INVERSE);
    kernel_complex_divide << <blocks, threads, 0, stream >> >(qframe, frame_resolution, static_cast<float>(n_frame_resolution));
  }

  cudaStreamSynchronize(stream);
}