#include "test.cuh"

float *make_sqrt_vec2(int vec_size)
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

void img2disk(std::string path, void* img, unsigned int size)
{
  FILE* fd_;

  if (fopen_s(&fd_, path.c_str(), "w+b") != 0)
    std::cout << "couldn't open file" << path << std::endl;
  else
  {
    if (fwrite(img, size, 1, fd_) != 1)
      std::cout << "couldn't write file to disk" << std::endl;
    fclose(fd_);
  }
}

cufftComplex* create_lens(unsigned int size_x, unsigned int size_y, float lambda, float z)
{
  // FIXME compute dynamically (nb threads)
  dim3 lthreads(16, 16);
  dim3 lblocks(size_x / 16, size_y / 16);
  cufftComplex *lens;
  cudaMalloc(&lens, size_x * size_y * sizeof(cufftComplex));
  kernel_quadratic_lens << <lblocks, lthreads >> >(lens, size_x, size_y, lambda, z);

  return lens;
}

void* fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect)
{
  // Sizes
  unsigned int pixel_size = q->get_frame_desc().width * q->get_frame_desc().height * nbimages;
  unsigned int complex_size = pixel_size * sizeof(cufftComplex);
  unsigned int short_size = pixel_size * sizeof(unsigned short);

  // Loaded images --> complex
  int threads = 512;
  unsigned int blocks = (pixel_size + threads - 1) / threads;

  // Hardware limit !!
  if (blocks >= 65536)
    blocks = 65535;

  cufftComplex* complex_input;
  cudaMalloc(&complex_input, complex_size);

  image_2_complex8 << <blocks, threads >> >(complex_input, (unsigned char*)q->get_last_images(nbimages), pixel_size, sqrt_vect);

  // Apply lens
  apply_quadratic_lens << <blocks, threads >> >(complex_input, pixel_size, lens, q->get_pixels());

  // FFT
  cufftHandle plan;
  cufftPlan3d(&plan, nbimages, q->get_frame_desc().width, q->get_frame_desc().height, CUFFT_C2C);
  cufftExecC2C(plan, complex_input, complex_input, CUFFT_FORWARD);

  // Complex --> real (unsigned short)
  unsigned short* real_output_gpu;
  cudaMalloc(&real_output_gpu, short_size);

  complex_2_module << <blocks, threads >> >(complex_input, real_output_gpu, pixel_size);

  // Write to disk
  unsigned short* real_output_cpu = (unsigned short*)malloc(short_size);
  cudaMemcpy(real_output_cpu, real_output_gpu, short_size, cudaMemcpyDeviceToHost);

  img2disk("atest_jeff.raw", real_output_cpu, short_size);

  // Free all
  cudaFree(complex_input);
  free(real_output_cpu);

  return real_output_gpu;
}





















/*
float *test_16(int nbimages, holovibes::Queue *q)
{
  std::cout << "test_16" << std::endl;
  //std::cout << "nb elt" << q->get_end_index() - q->get_start_index() << std::endl;
  float *img_gpu = make_contigous_float(q, nbimages);
  float *img_cpu = (float*)malloc(q->get_pixels() * sizeof(float) * nbimages);
  if (img_cpu)
    std::cout << "alloc ok" << std::endl;
   cudaMemcpy(img_cpu, img_gpu, q->get_pixels() * sizeof(float)* nbimages, cudaMemcpyDeviceToHost);
   img2disk("atest16.raw", img_cpu, q->get_pixels() * sizeof(float)* nbimages);
   return img_gpu;
}
*/