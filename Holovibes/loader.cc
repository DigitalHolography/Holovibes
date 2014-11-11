#include "stdafx.h"
#include "loader.hh"

ImageLoader::ImageLoader(std::string path, int nbimages, int bytedepth, int width, int height, holovibes::Queue *q)
{
  FILE *fd;
  fopen_s(&fd, path.c_str(), "r+b");
  void *img = malloc(width * height * bytedepth * nbimages);
  fread((void*)img, width * height * bytedepth, nbimages, fd);
  for (int i = 0; i < nbimages; i++)
  {
    q->enqueue((char*)img + (i * width * height * bytedepth), cudaMemcpyHostToDevice);
  }
  free(img);
}

ImageLoader::~ImageLoader()
{
}