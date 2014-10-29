#include "fft1.cuh"

void img2disk(std::string path, void* img, unsigned int size)
{
  FILE* fd_;

  void* cpu_img = malloc(size);
  if (cudaMemcpy(cpu_img, img, size, cudaMemcpyDeviceToHost) != CUDA_SUCCESS)
    std::cout << "couldn't copy mem" << std::endl;

  if (fopen_s(&fd_, path.c_str(), "w+b") != 0)
    std::cout << "couldn't open file" << path << std::endl;
  else
  {
    if (fwrite(cpu_img, size, 1, fd_) != 1)
      std::cout << "couldn't write file to disk" << std::endl;
    fclose(fd_);
  }
}

cufftComplex* create_lens(unsigned int size_x, unsigned int size_y, float lambda, float z)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);
  cufftComplex *lens;
  cudaMalloc(&lens, size_x * size_y * sizeof(cufftComplex));
  kernel_quadratic_lens << <lblocks, lthreads >> >(lens, size_x, size_y, lambda, z);

  return lens;
}

void fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan)
{
  // Sizes
  unsigned int pixel_size = q->get_frame_desc().width * q->get_frame_desc().height * nbimages;
  unsigned int complex_size = pixel_size * sizeof(cufftComplex);
  unsigned int short_size = pixel_size * sizeof(unsigned short);

  // Loaded images --> complex
  int threads = get_max_threads_1d();
  unsigned int blocks = (pixel_size + threads - 1) / threads;

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  cufftComplex* complex_input = make_contigous_complex(q, nbimages, sqrt_vect);

  // Apply lens
  apply_quadratic_lens <<<blocks, threads >> >(complex_input, pixel_size, lens, q->get_pixels());

  // FFT
 cufftExecC2C(plan, complex_input, complex_input, CUFFT_FORWARD);

  // Complex --> real (unsigned short)
  complex_2_module << <blocks, threads >> >(complex_input, result_buffer, pixel_size);
  //img2disk("at.raw", result_buffer, short_size);
  //std::cout << nbimages << std::endl;
  //getchar();
  //exit(0);

  // Free all
  cudaFree(complex_input);
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