# include "stft.cuh"

# include <cuda_runtime.h>
# include "hardware_limits.hh"
# include "tools.cuh"

void stft(
  cufftComplex*                   input,
  cufftComplex*                   lens,
  cufftComplex*                   stft_buf,
  cufftComplex*                   stft_dup_buf,
  cufftHandle                     plan2d,
  cufftHandle                     plan1d,
  const holovibes::Rectangle&     r,
  unsigned int&                   curr_elt,
  camera::FrameDescriptor&        desc,
  unsigned int                    nsamples,
  unsigned int                    pindex)
{
  unsigned int threads = 128;
  unsigned int blocks = desc.frame_res() / threads;
  
  if (blocks > get_max_blocks())
    blocks = get_max_blocks();
  
  // Apply lens on multiple frames.
  kernel_apply_lens << <blocks, threads >> >(input, desc.frame_res(), lens, desc.frame_res());

  cudaDeviceSynchronize();

  // FFT 2D
  cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);

  cudaDeviceSynchronize();

  if (curr_elt == nsamples)
  {
    // Remove first element and move all element on left
    cudaMemcpy(stft_buf, &(stft_buf[1]), sizeof(cufftComplex)* (nsamples * r.area() - 1), cudaMemcpyDeviceToDevice);
    --curr_elt;
  }
  // Do the ROI
  kernel_bursting_roi << <blocks, threads >> >(
    input,
    r.top_left.x,
    r.top_left.y,
    r.bottom_right.x,
    r.bottom_right.y,
    curr_elt,
    nsamples,
    desc.width,
    stft_buf);
  ++curr_elt;

  // FFT 1D
  cufftExecC2C(plan1d, stft_buf, stft_dup_buf, CUFFT_FORWARD);
  cudaDeviceSynchronize();

  // Reconstruct Roi
  kernel_reconstruct_roi << <blocks, threads >> >(
    stft_dup_buf,
    input,
    r.get_width(),
    r.get_height(),
    desc.width,
    pindex,
    nsamples);
}