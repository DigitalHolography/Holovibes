#include "test.cuh"

void img2disk(std::string path, void* img, unsigned int size)
{
  FILE* fd_;

  for (unsigned int i = 0; i < size; i++)
  {
    ((char*)img)[i] *= 50;
    //std::cout << i << std::endl;
  }

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
  unsigned int blocks = (q->get_size() * nbimages + threads - 1) / threads;

  if (blocks > 65536)
  {
    blocks = 65536;
  }

  void *img_gpu;

  cudaMalloc(&img_gpu, q->get_size() * nbimages);
  cufftComplex *result_fft = fft_3d(q, nbimages);
  //cufftComplex *result_fft_cpu = (cufftComplex*) malloc(sizeof (cufftComplex) * 10000);
  //cudaMemcpy(result_fft_cpu, result_fft, 10000, cudaMemcpyDeviceToHost);
  //for (int i = 0; i < 100; i++)
 // {
  //  std::cout << result_fft_cpu[i].x << std::endl;
 // }
  if (q->get_frame_desc().depth > 1)
  complex_2_module <<<blocks, threads >> >(result_fft, (unsigned short*)img_gpu, q->get_pixels() * nbimages);
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