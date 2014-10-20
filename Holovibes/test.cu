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
  unsigned char *img_gpu;

  cudaMalloc(&img_gpu, q->get_size() * nbimages);
  cudaMemset(img_gpu, 255, q->get_size() * nbimages);

  cufftComplex *result_fft = fft_3d(q, nbimages);

  complex_2_module << <blocks, threads >> >(result_fft, img_gpu, q->get_size() * nbimages);

  unsigned char *img_cpu = (unsigned char*)malloc(q->get_size() * nbimages);
  cudaMemcpy(img_cpu, img_gpu, q->get_size() *nbimages, cudaMemcpyDeviceToHost);
  img2disk("afft.raw", img_cpu, q->get_size() * nbimages);
}