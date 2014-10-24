#include "stdafx.h"
#include "fouriermanager.hh"

FourrierManager::FourrierManager(int p, int nbimages, float lambda, float dist, holovibes::Queue *q)
:p_(p),
nbimages_(nbimages_),
lambda_(lambda),
dist_(dist),
inputq_(q)
{
  camera::FrameDescriptor fd = inputq_->get_frame_desc();
  fd.depth = 2;
  outputq_ =  new holovibes::Queue(fd, inputq_->get_max_elts());
  cudaMalloc(&output_buffer_, q->get_pixels() * sizeof (unsigned short) * nbimages_);
  if (q->get_frame_desc().depth > 1)
    sqrt_vec_ = make_sqrt_vec(65535);
  else
    sqrt_vec_ = make_sqrt_vec(256);
  lens_ = create_lens(inputq_->get_frame_desc().width, inputq_->get_frame_desc().height, lambda_, dist_);
}




void FourrierManager::compute_hologram()
{
  fft_1(nbimages_, inputq_, lens_, sqrt_vec_, output_buffer_);
  unsigned short *img_p = output_buffer_ + (p_ * inputq_->get_pixels());
  outputq_->enqueue(img_p, cudaMemcpyDeviceToDevice);
}

holovibes::Queue *FourrierManager::get_queue()
{
  return outputq_;
}

FourrierManager::~FourrierManager()
{
  delete(outputq_);
  cudaFree(lens_);
  cudaFree(output_buffer_);
  cudaFree(sqrt_vec_);
}