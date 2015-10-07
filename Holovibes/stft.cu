# include "stft.cuh"

# include <cuda_runtime.h>
# include "hardware_limits.hh"
# include "tools.cuh"


void stft(
  cufftComplex*                   input,
  cufftComplex*                   lens,
  cufftComplex*                   stft_buf,
  cufftHandle                     plan2d,
  const holovibes::Rectangle&     r,
  unsigned int&                   curr_elt,
  unsigned int                    frame_resolution,
  unsigned int                    nframes)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = 128;
  unsigned int blocks = n_frame_resolution / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  // Apply lens on multiple frames.
  kernel_apply_lens <<<blocks, threads >>>(input, n_frame_resolution, lens, frame_resolution);

  cudaDeviceSynchronize();
  
  // FFT
  cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);
  
  // Do the ROI
  kernel_bursting_roi<<<blocks, threads>>>(
    input,
    r.top_left.x,
    r.top_left.y,
    r.bottom_right.x,
    r.bottom_right.y,
    curr_elt,
    nframes,
    2048,
    stft_buf); // USE desc.width

  cudaDeviceSynchronize();
  curr_elt = (++curr_elt) % nframes;
}