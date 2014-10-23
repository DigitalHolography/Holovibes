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
  unsigned int blocks = (q->get_pixels() * nbimages + threads - 1) / threads;
  unsigned int cube_size = q->get_pixels() * nbimages * sizeof(unsigned short);

  if (blocks > 65536)
    blocks = 65536;

  unsigned short* img_gpu;
  cudaMalloc(&img_gpu, cube_size);

  cufftComplex *result_fft = fft_3d(q, nbimages); // return a cufftComplex result of the fft3d of nbimages

  complex_2_module <<<blocks, threads >> >(result_fft, img_gpu, q->get_pixels() * nbimages); // convert the complex im

  unsigned short* img_cpu = (unsigned short*)malloc(cube_size);
  cudaMemcpy(img_cpu, img_gpu, cube_size, cudaMemcpyDeviceToHost);

  img2disk("afft.raw", img_cpu, cube_size);
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