#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"
#include "fouriermanager.hh"

int main(int argc, const char* argv[])
{
  camera::FrameDescriptor desc;
  desc.depth = 1;
  desc.height = 2048;
  desc.width = 2048;
  desc.pixel_size = 1;
  FILE *fd;

  // Loading images
  int images_2_read = 16;
  fopen_s(&fd, "2phase_rawfrog.raw", "r+b");
  unsigned char *img = (unsigned char*)malloc(images_2_read * 2048 * 2048);
  fread((void*)img, 2048 * 2048, images_2_read, fd);

  holovibes::Queue *q = new holovibes::Queue(desc, 20);
  for (int i = 0; i < images_2_read; i++)
  {
    q->enqueue((void*)&img[i * desc.width * desc.height * desc.depth], cudaMemcpyHostToDevice);
  }
  //
  holovibes::FourrierManager fm = holovibes::FourrierManager(8, 10, 535.0e-9f, 1.36f, *q);

  cudaEvent_t start, stop;
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


  fm.compute_hologram();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  printf("Time for the kernel: %f ms\n", time);
  //std::cout << get_max_blocks() << std::endl; 


  void* img_gpu = fm.get_queue()->get_last_images(1);

  img2disk("at.raw", img_gpu, fm.get_queue()->get_size());
  getchar();
  return 0;
}
