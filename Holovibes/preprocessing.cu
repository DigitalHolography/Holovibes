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
  int threads = get_max_threads_1d();
  int blocks = (q->get_pixels() * nbimages + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  int vec_size_pix = q->get_pixels() * nbimages;
  int vec_size_byt = q->get_size() * nbimages;
  int img_byte = q->get_size();
  int contigous_elts = 0;

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
    cudaMemcpy(contigous, q->get_start(), contigous_elts * img_byte, cudaMemcpyDeviceToDevice);

    // Copy the contiguous elements left of the beginning of the queue into buffer
    cudaMemcpy(contigous + contigous_elts * img_byte, q->get_buffer(), (nbimages - contigous_elts) * img_byte, cudaMemcpyDeviceToDevice);

    if (q->get_frame_desc().depth > 1)
      image_2_complex16 <<<blocks, threads >> >(output, (unsigned short*)contigous, vec_size_pix, sqrt_vec);
    else
      image_2_complex8 <<<blocks, threads >> >(output, contigous, vec_size_pix, sqrt_vec);

    cudaFree(contigous);
    return output;
  }
  else
  {
    if (q->get_frame_desc().depth > 1)
    {
      // fix me
      img2disk("at.raw", q->get_start(), q->get_size() * nbimages);
      //getchar();
      //exit(0);
      //fixme
      void *my_image;//fix me
      cudaMalloc(&my_image, q->get_size() * nbimages); //f
      image_2_complex16 << <blocks, threads >> >(output, (unsigned short*)q->get_start(), vec_size_pix, sqrt_vec);
      complex_2_module << <blocks, threads >> >(output, (unsigned short*)my_image, q->get_pixels() * nbimages);//f
      img2disk("ab.raw",my_image, q->get_size() * nbimages);//f
      exit(0); //f`


      std::cout << " in contuigousd 16" << std::endl;
    }
    else
    {
      image_2_complex8 << <blocks, threads >> >(output, (unsigned char*)q->get_start(), vec_size_pix, sqrt_vec);
    }
    return output;
  }
  
}
