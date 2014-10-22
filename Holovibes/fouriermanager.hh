#ifndef FOURIERMANAGER_HH
#define FOURRIERMANAGER_HH

#include "queue.hh"
#include "fourier_computing.cuh"

class FourrierManager
{
public:
  FourrierManager(int p, int nbimages, float lambda, float dist,holovibes::Queue *q);
  ~FourrierManager();
  void *get_image();

private:
  void *compute_image_vector();
  void gpu_vec_extract(unsigned char *gpu_vec);
  int threads_;
  int bytedepth_;
  int p_;
  int nbimages_;
  float lambda_;
  float dist_;
  holovibes::Queue* outputq_;
  holovibes::Queue* inputq_;


};








#endif