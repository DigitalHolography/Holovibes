#include "preprocessing.cuh"

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

cufftComplex *make_contigous_complex(holovibes::Queue *q, int nbimages)
{
  float *sqrt_vec;
  if (q->get_frame_desc().depth == 1)
    sqrt_vec = make_sqrt_vec(256);
  else
    sqrt_vec = make_sqrt_vec(65536);//think about free it
  int threads = 512;
  int blocks = (q->get_size() * nbimages + 511) / 512;
  if (blocks > 65536)
  {
    blocks = 65536;
  }
  int vec_size_pix = q->get_pixels() * nbimages;
  int vec_size_byt = q->get_size() * nbimages;
  int img_byte = q->get_size();
  int contigous_elts = 0;
  cufftComplex *output;
  cudaMalloc(&output, vec_size_pix * sizeof(cufftComplex));
  int index = (q->get_start_index() + q->get_current_elts() - nbimages) % q->get_max_elts();

  if (q->get_max_elts() - index < nbimages)
    contigous_elts = q->get_max_elts() - nbimages;
  else
    contigous_elts = nbimages;

  if (contigous_elts < nbimages)
  {
    unsigned char *contigous;
    cudaMalloc(&contigous, vec_size_byt);
    cudaMemcpy(contigous, q->get_last_images(nbimages), contigous_elts * img_byte, cudaMemcpyDeviceToDevice);
    cudaMemcpy(contigous + contigous_elts * img_byte, q->get_buffer(), (nbimages - contigous_elts) * img_byte, cudaMemcpyDeviceToDevice);
    if (q->get_frame_desc().depth > 1)
      image_2_complex16 << <blocks, threads >> >(output, (unsigned short*)contigous, vec_size_pix, sqrt_vec);
    else
      image_2_complex8 << <blocks, threads >> >(output, contigous, vec_size_pix, sqrt_vec);

    return output;
  }
  else
  {
    std::cout << "hey" << std::endl;
    if (q->get_frame_desc().depth > 1)
    {
      image_2_complex16 << <blocks, threads >> >(output, (unsigned short*)q->get_last_images(nbimages), vec_size_pix, sqrt_vec);
      std::cout << "in contigous 16 bit" << std::endl;
    }
    else
    {
      image_2_complex8 << <blocks, threads >> >(output, (unsigned char*)q->get_last_images(nbimages), vec_size_pix, sqrt_vec);
      std::cout << "in contigous 8 bit" << std::endl;
    }
    return output;
  }
  
}
