#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"
#include "test.cuh"

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

  holovibes::Queue *q = new holovibes::Queue(desc, 50);
  for (int i = 0; i < images_2_read; i++)
  {
    q->enqueue((void*)&img[i * desc.width * desc.height * desc.depth], cudaMemcpyHostToDevice);
  }

  test_fft(16, q);

  return 0;
}
