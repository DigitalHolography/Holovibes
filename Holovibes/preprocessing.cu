#include "preprocessing.cuh"

float *make_contigous_float(holovibes::Queue *q, int nbimages)
{
  int threads = 512;
  unsigned int blocks = (q->get_size() * nbimages + 511) / 512;
  if (blocks > 65536)
  {
    blocks = 65536;
  }

  std::cout << "blocks :" << blocks << std::endl;
  int contigous_elts = 0;
  cufftReal *output;
  cudaMalloc(&output, q->get_pixels() * nbimages * sizeof(cufftReal));
  int index = (q->get_start_index() + q->get_current_elts() - nbimages) % q->get_max_elts();

  if (q->get_max_elts() - index < nbimages)
    contigous_elts = q->get_max_elts() - nbimages;
  else
    contigous_elts = nbimages;
  if (contigous_elts < nbimages)
  {
    unsigned char *contigous;
    cudaMalloc(&contigous, q->get_size() * nbimages);
    cudaMemcpy(contigous, q->get_last_images(nbimages), contigous_elts * q->get_size(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(contigous + contigous_elts * q->get_size(), q->get_buffer(), (nbimages - contigous_elts) * q->get_size(), cudaMemcpyDeviceToDevice);
    if (q->get_frame_desc().get_byte_depth() > 1)
      image_2_float <<<blocks, threads >>>(output,(unsigned short*) contigous, q->get_pixels() * nbimages);
    else
      image_2_float << <blocks, threads >> >(output, contigous, q->get_pixels() * nbimages);

    return output;
  }
  else
  {
    std::cout << "contigous: " <<contigous_elts << std::endl;
    if (q->get_frame_desc().get_byte_depth() > 1)
      image_2_float <<<blocks, threads >> >(output, (unsigned short*)q->get_last_images(nbimages), q->get_pixels() * nbimages);
    else
      image_2_float <<<blocks, threads >> >(output, (unsigned char*) q->get_last_images(nbimages), q->get_pixels() * nbimages);
    return output;
  }
}

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
  unsigned char *contigous;
  cudaMalloc(&contigous, q->get_size() * nbimages);
  float *sqrt_vec;
  if (q->get_frame_desc().get_byte_depth() == 1)
    sqrt_vec = make_sqrt_vec(256);
  else
    sqrt_vec = make_sqrt_vec(65536);//think about free it
  int threads = 512;
  int blocks = (q->get_size() * nbimages + 511) / 512;
  if (blocks > 65536)
  {
    blocks = 65536;
  }
  int contigous_elts = 0;
  cufftComplex *output;
  cudaMalloc(&output, q->get_pixels() * nbimages * sizeof(cufftComplex));
  int index = (q->get_start_index() + q->get_current_elts() - nbimages) % q->get_max_elts();

  if (q->get_max_elts() - index < nbimages)
    contigous_elts = q->get_max_elts() - nbimages;
  else
    contigous_elts = nbimages;
  if (contigous_elts < nbimages)
  {
    cudaMemcpy(contigous, q->get_last_images(nbimages), contigous_elts * q->get_size(), cudaMemcpyDeviceToDevice);
    cudaMemcpy(contigous + contigous_elts * q->get_size(), q->get_buffer(), (nbimages - contigous_elts) * q->get_size(), cudaMemcpyDeviceToDevice);
  }
  else
  {
    std::cout << "image naturally contigous" << std::endl;
    cudaMemcpy(contigous, q->get_last_images(nbimages), nbimages * q->get_size(), cudaMemcpyDeviceToDevice);
  }

    if (q->get_frame_desc().get_byte_depth() > 1)
      image_2_complex <<<blocks, threads >> >(output, (unsigned short*)contigous, q->get_pixels() * nbimages, sqrt_vec);
    else
      image_2_complex <<<blocks, threads >> >(output, (unsigned char*)contigous, q->get_pixels() * nbimages, sqrt_vec);
    return output;
  
}
