#include "preprocessing.cuh"
#include "fft1.cuh"

float *make_sqrt_vec(int vec_size)
{
  float *vec = (float*)malloc(sizeof(float)* vec_size);
  for (int i = 0; i < vec_size; i++)
    vec[i] = sqrt(i);
  float *vec_gpu;
  cudaMalloc(&vec_gpu, sizeof(float)* vec_size);
  cudaMemcpy(vec_gpu, vec, sizeof(float)* vec_size, cudaMemcpyHostToDevice);
  free(vec);
  return vec_gpu;
}

cufftComplex *make_contigous_complex(holovibes::Queue *q, int nbimages, float *sqrt_vec)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (q->get_pixels() * nbimages + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;
  blocks = 65535;

  unsigned int vec_size_pix = q->get_pixels() * nbimages;
  unsigned int vec_size_byt = q->get_size() * nbimages;
  unsigned int img_byte = q->get_size();
  unsigned int contigous_elts = 0;

  cufftComplex *output;
  cudaMalloc(&output, vec_size_pix * sizeof(cufftComplex));

  if (q->get_start_index() + nbimages <= q->get_max_elts())
    contigous_elts = nbimages;
  else
    contigous_elts = q->get_max_elts() - q->get_start_index();

  if (contigous_elts < nbimages)
  {
    unsigned char *contigous;
    cudaMalloc(&contigous, vec_size_byt);

    // Copy contiguous elements of the end of the queue into buffer
    if (cudaMemcpy(contigous, q->get_start(), contigous_elts * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cout << "non contiguous memcpy failed" << std::endl;

    // Copy the contiguous elements left of the beginning of the queue into buffer
    if (cudaMemcpy(contigous + contigous_elts * img_byte, q->get_buffer(), (nbimages - contigous_elts) * img_byte, cudaMemcpyDeviceToDevice) != CUDA_SUCCESS)
      std::cout << "non contiguous memcpy failed" << std::endl;

    if (q->get_frame_desc().depth > 1)
      image_2_complex16 <<<blocks, threads >> >(output, (unsigned short*)contigous, vec_size_pix, sqrt_vec);
    else
      image_2_complex8 <<<blocks, threads >> >(output, contigous, vec_size_pix, sqrt_vec);

    //std::cout << "not contiguous" << std::endl;

    //img2disk("atest.raw", contigous, vec_size_byt);
    //exit(0);
    if(cudaFree(contigous) != CUDA_SUCCESS)
      std::cout << "non contiguous free failed" << std::endl;

    return output;
  }
  else
  {
    if (q->get_frame_desc().depth > 1)
    {
      image_2_complex16 << <blocks, threads >> >(output, (unsigned short*)q->get_start(), vec_size_pix, sqrt_vec);
      //std::cout << " in contuigousd 16" << std::endl;
    }
    else
    {
      image_2_complex8 << <blocks, threads >> >(output, (unsigned char*)q->get_start(), vec_size_pix, sqrt_vec);
    }
    return output;
  }
  
}
