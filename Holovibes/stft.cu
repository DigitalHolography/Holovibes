#include "stft.cuh"
#include "hardware_limits.hh"
#include "geometry.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"

void stft(
  cufftComplex*                   input,
  const cufftComplex*             gpu_queue,
  cufftComplex*                   stft_buf,
  cufftComplex*                   stft_dup_buf,
  const cufftHandle               plan2d,
  const cufftHandle               plan1d,
  const holovibes::Rectangle&     r,
  unsigned int&                   curr_elt,
  const camera::FrameDescriptor&  desc,
  unsigned int                    nsamples,
  unsigned int                    stft_level,
  cudaStream_t                    stream)
{
  unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(desc.frame_res(), threads);

 /* // Apply lens on multiple frames.
  kernel_apply_lens << <blocks, threads, 0, stream >> >(input, desc.frame_res(), lens, desc.frame_res());

  cudaStreamSynchronize(stream);

  // FFT 2D
  cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);
  cudaStreamSynchronize(stream);*/

  if (!r.area())
    return;

/*  if (curr_elt == nsamples)
  {
    // Remove first element and move all element on left
    cudaMemcpyAsync(stft_buf,
      &(stft_buf[1]),
      sizeof(cufftComplex)* (nsamples * r.area() - 1),
      cudaMemcpyDeviceToDevice,
      stream);
    --curr_elt;
  }*/

  // Do the ROI
/*  kernel_bursting_roi << <blocks, threads, 0, stream >> >(
    input,
    r.top_left.x,
    r.top_left.y,
    r.bottom_right.x,
    r.bottom_right.y,
    curr_elt,
    nsamples,
    desc.width,
    desc.width * desc.height,
    stft_buf);
  ++curr_elt;*/

  // FFT 1D
  cufftExecC2C(plan1d, stft_buf, stft_dup_buf, CUFFT_FORWARD);
  cudaStreamSynchronize(stream);
}

void stft_recontruct(
  cufftComplex*                   output,
  cufftComplex*                   stft_dup_buf,
  const holovibes::Rectangle      r,
  const camera::FrameDescriptor&  desc,
  const unsigned int              reconstruct_width,
  const unsigned int              reconstruct_height,
  const unsigned int              pindex,
  const unsigned int              nsamples,
  cudaStream_t stream)
{
  unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(desc.frame_res(), threads);

  if (!r.area())
    return;
  // Reconstruct Roi
/*  kernel_reconstruct_roi << <blocks, threads, 0, stream >> >(
    stft_dup_buf,
    output,
    r.get_width(),
    r.get_height(),
    desc.width,
    reconstruct_width,
    reconstruct_height,
    pindex,
    nsamples);*/
}