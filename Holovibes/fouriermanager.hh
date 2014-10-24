#ifndef FOURIERMANAGER_HH
#define FOURRIERMANAGER_HH

#include "queue.hh"
#include "fourier_computing.cuh"
#include "test.cuh"

class FourrierManager
{
public:
  FourrierManager(int p, int nbimages, float lambda, float dist,holovibes::Queue *q);
  FourrierManager();
  ~FourrierManager();
  void compute_hologram();
  holovibes::Queue *get_queue();

private:
  cufftComplex *lens_;
  float *sqrt_vec_; //gpu_vec
  unsigned short *output_buffer_;
  int threads_;
  int bytedepth_;
  int p_;
  int nbimages_;
  float lambda_;
  float dist_;
  holovibes::Queue *outputq_;
  holovibes::Queue* inputq_;


};








#endif