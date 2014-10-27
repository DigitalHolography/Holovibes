#ifndef LOADER_HH
#define LOADER_HH

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"
#include "fft1.cuh"

class ImageLoader
{
public:
  ImageLoader(std::string path, int nbimages, int bytedepth, int width, int height, holovibes::Queue *q);
  ~ImageLoader();
private:

};



#endif