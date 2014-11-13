#include "preprocessing.cuh"

#include "hardware_limits.hh"
#include "tools.cuh"

void make_sqrt_vect(float* out, unsigned short n)
{
  float* vect = new float[n]();

  for (size_t i = 0; i < n; ++i)
    vect[i] = sqrtf(static_cast<float>(i));

  cudaMemcpy(out, vect, sizeof(float) * n, cudaMemcpyHostToDevice);

  delete[] vect;
}

cufftComplex *make_contiguous_complex(
  holovibes::Queue& q,
  unsigned int nbimages,
  float *sqrt_vec)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (q.get_pixels() * nbimages + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  unsigned int vec_size_pix = q.get_pixels() * nbimages;
  size_t vec_size_byte = q.get_size() * nbimages;
  size_t img_byte = q.get_size();
  unsigned int contiguous_elts = 0;

  cufftComplex *output;
  cudaMalloc(&output, vec_size_pix * sizeof(cufftComplex));

  if (q.get_start_index() + nbimages <= q.get_max_elts())
    contiguous_elts = nbimages;
  else
    contiguous_elts = q.get_max_elts() - q.get_start_index();

  if (contiguous_elts < nbimages)
  {
    unsigned char *contiguous;
    cudaMalloc(&contiguous, vec_size_byte);

    // Copy contiguous elements of the end of the queue into buffer
    if (cudaMemcpy(contiguous, q.get_start(), contiguous_elts * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cerr << "non contiguous memcpy failed" << std::endl;

    // Copy the contiguous elements left of the beginning of the queue into buffer
    if (cudaMemcpy(contiguous + contiguous_elts * img_byte, q.get_buffer(), (nbimages - contiguous_elts) * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cerr << "non contiguous memcpy failed" << std::endl;

    if (q.get_frame_desc().depth > 1)
      image_2_complex16 <<<blocks, threads >>>(output, (unsigned short*)contiguous, vec_size_pix, sqrt_vec);
    else
      image_2_complex8 <<<blocks, threads >>>(output, contiguous, vec_size_pix, sqrt_vec);

    if (cudaFree(contiguous) != CUDA_SUCCESS)
      std::cerr << "non contiguous free failed" << std::endl;
  }
  else
  {
    if (q.get_frame_desc().depth > 1)
    {
      image_2_complex16 <<<blocks, threads >>>(output, (unsigned short*)q.get_start(), vec_size_pix, sqrt_vec);
    }
    else
    {
      image_2_complex8 <<<blocks, threads >>>(output, (unsigned char*)q.get_start(), vec_size_pix, sqrt_vec);
    }
  }

  return output;
}
