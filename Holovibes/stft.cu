# include "stft.cuh"

# include <cuda_runtime.h>
# include "hardware_limits.hh"
#include "tools.cuh"

void stft(
  cufftComplex* input,
  cufftComplex* lens,
  cufftComplex* stft_buf,
  cufftHandle   plan2d,
  unsigned int frame_resolution,
  unsigned int nframes)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = n_frame_resolution / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  // Apply lens on multiple frames.
  kernel_apply_lens <<<blocks, threads >>>(input, n_frame_resolution, lens, frame_resolution);


  cudaDeviceSynchronize();
  
  // FFT
  cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);
  
  cudaDeviceSynchronize();
}