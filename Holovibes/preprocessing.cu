#include "preprocessing.cuh"

#include "hardware_limits.hh"
#include "tools.cuh"

/*! \brief Precompute the sqrt q sqrt vector of values in
* range 0 to n.
*
* \param n Number of values to compute.
* \param output Array of the sqrt values form 0 to n - 1,
* this array should have size greater or equal to n.
*/
void make_sqrt_vect(float* out, unsigned short n)
{
  float* vect = new float[n]();

  for (size_t i = 0; i < n; ++i)
    vect[i] = sqrtf(static_cast<float>(i));

  cudaMemcpy(out, vect, sizeof(float) * n, cudaMemcpyHostToDevice);

  delete[] vect;
}

/*! \brief Ensure the contiguity of images extracted from
 * the queue for any further processing.
 * This function also compute the sqrt value of each pixel of images.
 * 
 * \param input the device queue from where images should be taken
 * to be processed.
 * \param output A bloc made of n contigus images requested 
 * to the function.
 * \param n Number of images to ensure contiguity.
 * \param sqrt_array Array of the sqrt values form 0 to 65535
 * in case of 16 bit images or from 0 to 255 in case of 
 * 8 bit images.
 *
 *
 *\note This function can be improved by specifying
 * img8_to_complex or img16_to_complex in the pipeline to avoid
 * branch conditions. But it is no big deal.
 * Otherwise, the convert function are not called outside because
 * this function would need an unsigned short buffer that is unused
 * anywhere else.
 */
void make_contiguous_complex(
  holovibes::Queue& input,
  cufftComplex* output,
  unsigned int n,
  const float* sqrt_array)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (input.get_pixels() * n + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  const unsigned int frame_resolution = input.get_pixels();
  const camera::FrameDescriptor& frame_desc = input.get_frame_desc();

  if (input.get_start_index() + n <= input.get_max_elts())
  {
    const unsigned int n_frame_resolution = frame_resolution * n;
    /* Contiguous case. */
    if (frame_desc.depth > 1)
    {
      img16_to_complex<<<blocks, threads>>>(
        output,
        static_cast<unsigned short*>(input.get_start()),
        n_frame_resolution,
        sqrt_array);
    }
    else
    {
      img8_to_complex<<<blocks, threads>>>(
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
      img16_to_complex<<<blocks, threads>>>(
        output,
        static_cast<unsigned short*>(input.get_start()),
        contiguous_elts_res,
        sqrt_array);

      // Convert the contiguous elements left (at the beginning of queue).
      img16_to_complex<<<blocks, threads>>>(
        output + contiguous_elts_res,
        static_cast<unsigned short*>(input.get_buffer()),
        left_elts_res,
        sqrt_array);
    }
    else
    {
      // Convert contiguous elements (at the end of the queue).
      img8_to_complex<<<blocks, threads>>>(
        output,
        static_cast<unsigned char*>(input.get_start()),
        contiguous_elts_res,
        sqrt_array);

      // Convert the contiguous elements left (at the beginning of queue).
      img8_to_complex<<<blocks, threads>>>(
        output + contiguous_elts_res,
        static_cast<unsigned char*>(input.get_buffer()),
        left_elts_res,
        sqrt_array);
    }
  }
}
