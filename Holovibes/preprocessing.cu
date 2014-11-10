#include "preprocessing.cuh"
#include "fft1.cuh"
#include <iostream>

float *make_sqrt_vect(int vect_size)
{
  float* vect = new float[vect_size]();

  for (int i = 0; i < vect_size; i++)
    vect[i] = sqrtf(i);

  float* vect_gpu;
  cudaMalloc(&vect_gpu, sizeof(float) * vect_size);
  cudaMemcpy(vect_gpu, vect, sizeof(float) * vect_size, cudaMemcpyHostToDevice);

  delete[] vect;

  return vect_gpu;
}

cufftComplex *make_contiguous_complex(holovibes::Queue *q, unsigned int nbimages, float *sqrt_vec)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (q->get_pixels() * nbimages + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  unsigned int vec_size_pix = q->get_pixels() * nbimages;
  size_t vec_size_byt = q->get_size() * nbimages;
  size_t img_byte = q->get_size();
  unsigned int contiguous_elts = 0;

  cufftComplex *output;
  cudaMalloc(&output, vec_size_pix * sizeof(cufftComplex));

  if (q->get_start_index() + nbimages <= q->get_max_elts())
    contiguous_elts = nbimages;
  else
    contiguous_elts = q->get_max_elts() - q->get_start_index();

  if (contiguous_elts < nbimages)
  {
    unsigned char *contiguous;
    cudaMalloc(&contiguous, vec_size_byt);

    // Copy contiguous elements of the end of the queue into buffer
    if (cudaMemcpy(contiguous, q->get_start(), contiguous_elts * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cerr << "non contiguous memcpy failed" << std::endl;

    // Copy the contiguous elements left of the beginning of the queue into buffer
    if (cudaMemcpy(contiguous + contiguous_elts * img_byte, q->get_buffer(), (nbimages - contiguous_elts) * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cerr << "non contiguous memcpy failed" << std::endl;

    if (q->get_frame_desc().depth > 1)
      image_2_complex16 <<<blocks, threads >> >(output, (unsigned short*)contiguous, vec_size_pix, sqrt_vec);
    else
      image_2_complex8 <<<blocks, threads >> >(output, contiguous, vec_size_pix, sqrt_vec);

    if (cudaFree(contiguous) != CUDA_SUCCESS)
      std::cerr << "non contiguous free failed" << std::endl;
  }
  else
  {
    if (q->get_frame_desc().depth > 1)
    {
      image_2_complex16 << <blocks, threads >> >(output, (unsigned short*)q->get_start(), vec_size_pix, sqrt_vec);
    }
    else
    {
      image_2_complex8 << <blocks, threads >> >(output, (unsigned char*)q->get_start(), vec_size_pix, sqrt_vec);
    }
  }

  return output;
}
