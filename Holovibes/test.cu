#include "test.cuh"

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

void test_fft(int nbimages, holovibes::Queue *q)
{
  int threads = 512;
  int blocks = (q->get_size() * nbimages + 511) / 512;
  void *img_gpu;

  cudaMalloc(&img_gpu, q->get_size() * nbimages);
  cufftComplex *result_fft = fft_3d(q, nbimages);
  if (q->get_frame_desc().depth > 1)
    complex_2_module << <blocks, threads >> >(result_fft, (unsigned short*)img_gpu, q->get_pixels() * nbimages);
  else
    complex_2_module << <blocks, threads >> >(result_fft, (unsigned char*)img_gpu, q->get_pixels() * nbimages);

  void *img_cpu = (void*)malloc(q->get_size() * nbimages);
  cudaMemcpy(img_cpu, img_gpu, q->get_size() *nbimages, cudaMemcpyDeviceToHost);
  img2disk("afft.raw", img_cpu, q->get_size() * nbimages);
}

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