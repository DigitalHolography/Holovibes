#include "preprocessing.cuh"
#include "hardware_limits.hh"
#include "tools_conversion.cuh"

void make_sqrt_vect(float* out, const unsigned short n)
{
  float* vect = new float[n]();

  for (size_t i = 0; i < n; ++i)
    vect[i] = sqrtf(static_cast<float>(i));

  cudaMemcpy(out, vect, sizeof(float)* n, cudaMemcpyHostToDevice);

  delete[] vect;
}

void make_contiguous_complex(
  holovibes::Queue& input,
  cufftComplex* output,
  const unsigned int n,
  const float* sqrt_array)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (input.get_pixels() * n + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  const unsigned int frame_resolution = input.get_pixels();
  const camera::FrameDescriptor& frame_desc = input.get_frame_desc();

  cudaMemcpy(output + frame_resolution,
    output,
    sizeof(cufftComplex)* (n - 1) * frame_resolution,
    cudaMemcpyDeviceToDevice);

  if (frame_desc.depth > 1)
  {
    img16_to_complex << <blocks, threads >> >(
      output,
      static_cast<unsigned short*>(input.get_start()),
      frame_resolution,
      sqrt_array);
  }
  else
  {
    img8_to_complex << <blocks, threads >> >(
      output,
      static_cast<unsigned char*>(input.get_start()),
      frame_resolution,
      sqrt_array);
  }
}