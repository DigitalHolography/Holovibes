#include "stdafx.h"
#include "fouriermanager.hh"

FourrierManager::FourrierManager(int p, int nbimages, float lambda, float dist, holovibes::Queue *q)
:inputq_(q),
lambda_(lambda),
nbimages_(nbimages_),
dist_(dist)
{
  int count;
  cudaGetDeviceCount(&count);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, count);
  threads_ = prop.maxThreadsPerBlock;
  bytedepth_ = q->get_frame_desc().depth;
  outputq_ = new holovibes::Queue(q->get_frame_desc(), q->get_max_elts());
}




void *FourrierManager::get_image()
{
  void* t = NULL;
  gpu_vec_extract((unsigned char*)compute_image_vector());
  outputq_->dequeue(t);
  return t;
}

void *FourrierManager::compute_image_vector()
{
  void *gpu_vector;
  cudaMalloc(&gpu_vector, inputq_->get_pixels() * nbimages_ * sizeof(unsigned short));
  int blocks = (inputq_->get_pixels() * nbimages_ + threads_ - 1) / threads_;
  cufftComplex *result_fft = fft_3d(inputq_, nbimages_);
  complex_2_modul_call(result_fft, (unsigned short*)gpu_vector, inputq_->get_pixels() * nbimages_, blocks, threads_);
  return gpu_vector;
}

void FourrierManager::gpu_vec_extract(unsigned char *gpu_vec)
{
  void *to_eq = gpu_vec + (p_ * inputq_->get_size());
  outputq_->enqueue(to_eq, cudaMemcpyHostToDevice);
  cudaFree(gpu_vec);
}












FourrierManager::~FourrierManager()
{
  delete(outputq_);
}