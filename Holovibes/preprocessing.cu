#include "preprocessing.cuh"
#include "hardware_limits.hh"
#include "tools.hh"
#include "tools_conversion.cuh"

void make_sqrt_vect(float* out,
  const unsigned short n,
  cudaStream_t stream)
{
  float* vect = new float[n]();

  for (size_t i = 0; i < n; ++i)
    vect[i] = sqrtf(static_cast<float>(i));

  cudaMemcpyAsync(out, vect, sizeof(float)* n, cudaMemcpyHostToDevice, stream);

  delete[] vect;
}

void make_contiguous_complex(
  holovibes::Queue& input,
  cufftComplex* output,
  const unsigned int n,
  const float* sqrt_array,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(input.get_pixels() * n, threads);

  const unsigned int frame_resolution = input.get_pixels();
  const camera::FrameDescriptor& frame_desc = input.get_frame_desc();

  if (input.get_start_index() + n <= input.get_max_elts())
  {
    const unsigned int n_frame_resolution = frame_resolution * n;
    /* Contiguous case. */
    if (frame_desc.depth > 1)
    {
      img16_to_complex << <blocks, threads, 0, stream >> >(
        output,
        static_cast<unsigned short*>(input.get_start()),
        n_frame_resolution,
        sqrt_array);
    }
    else
    {
      img8_to_complex << <blocks, threads, 0, stream >> >(
        output,
        static_cast<unsigned char*>(input.get_start()),
        n_frame_resolution,
        sqrt_array);
    }
  }
  else
  {
    const unsigned int contiguous_elts = input.get_max_elts() - input.get_start_index();
    const unsigned int contiguous_elts_res = frame_resolution * contiguous_elts;
    const unsigned int left_elts = n - contiguous_elts;
    const unsigned int left_elts_res = frame_resolution * left_elts;

    if (frame_desc.depth > 1)
    {
      // Convert contiguous elements (at the end of the queue).
      img16_to_complex << <blocks, threads, 0, stream >> >(
        output,
        static_cast<unsigned short*>(input.get_start()),
        contiguous_elts_res,
        sqrt_array);

      // Convert the contiguous elements left (at the beginning of queue).
      img16_to_complex << <blocks, threads, 0, stream >> >(
        output + contiguous_elts_res,
        static_cast<unsigned short*>(input.get_buffer()),
        left_elts_res,
        sqrt_array);
    }
    else
    {
      // Convert contiguous elements (at the end of the queue).
      img8_to_complex << <blocks, threads, 0, stream >> >(
        output,
        static_cast<unsigned char*>(input.get_start()),
        contiguous_elts_res,
        sqrt_array);

      // Convert the contiguous elements left (at the beginning of queue).
      img8_to_complex << <blocks, threads, 0, stream >> >(
        output + contiguous_elts_res,
        static_cast<unsigned char*>(input.get_buffer()),
        left_elts_res,
        sqrt_array);
    }
  }
}