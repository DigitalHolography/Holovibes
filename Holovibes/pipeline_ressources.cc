#include "stdafx.h"

#include <cuda_runtime.h>
#include <cassert>

#include "pipeline_ressources.hh"
#include "preprocessing.cuh"

namespace holovibes
{
  PipelineRessources::PipelineRessources(
    Queue& input,
    Queue& output,
    unsigned short n)
    : input_(input)
    , output_(output)
    , gpu_sqrt_vector_(nullptr)
    , gpu_pbuffer_(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , gpu_lens_(nullptr)
    , gpu_output_frame_(nullptr)
  {
    new_gpu_sqrt_vector(sqrt_vector_size);
    new_gpu_pbuffer(n);
    new_plan3d(n);
    new_plan2d();
    new_gpu_lens();
  }

  PipelineRessources::~PipelineRessources()
  {
    delete_gpu_sqrt_vector();
    delete_gpu_pbuffer();
    delete_plan3d();
    delete_plan2d();
    delete_gpu_lens();
  }

  /* Public methods */

  Queue& PipelineRessources::get_input_queue()
  {
    return input_;
  }

  Queue& PipelineRessources::get_output_queue()
  {
    return output_;
  }

  float* PipelineRessources::get_sqrt_vector() const
  {
    return gpu_sqrt_vector_;
  }

  unsigned short* PipelineRessources::get_pbuffer()
  {
    return gpu_pbuffer_;
  }

  cufftHandle PipelineRessources::get_plan3d()
  {
    return plan3d_;
  }

  cufftHandle PipelineRessources::get_plan2d()
  {
    return plan2d_;
  }

  cufftComplex* PipelineRessources::get_lens()
  {
    return gpu_lens_;
  }

  unsigned short*& PipelineRessources::get_output_frame_ptr()
  {
    return gpu_output_frame_;
  }

  void PipelineRessources::update_n_parameter(unsigned short n)
  {
    assert(n != 0 && "n must be strictly positive");
    delete_gpu_pbuffer();
    delete_plan3d();
    new_gpu_pbuffer(n);
    new_plan3d(n);
  }

  /* Private methods */

  void PipelineRessources::new_gpu_sqrt_vector(unsigned short n)
  {
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float) * n);
    make_sqrt_vect(gpu_sqrt_vector_, n);
  }

  void PipelineRessources::delete_gpu_sqrt_vector()
  {
    cudaFree(gpu_sqrt_vector_);
  }

  void PipelineRessources::new_gpu_pbuffer(unsigned short n)
  {
    assert(cudaMalloc/*<unsigned short>*/(&gpu_pbuffer_,
      sizeof(unsigned short) * input_.get_pixels() * n) == CUDA_SUCCESS);
  }

  void PipelineRessources::delete_gpu_pbuffer()
  {
    cudaFree(gpu_pbuffer_);
  }

  void PipelineRessources::new_plan3d(unsigned short n)
  {
    cufftPlan3d(
      &plan3d_,
      n,                              // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);
  }

  void PipelineRessources::delete_plan3d()
  {
    cufftDestroy(plan3d_);
  }

  void PipelineRessources::new_plan2d()
  {
    cufftPlan2d(
      &plan2d_,
      input_.get_frame_desc().width,
      input_.get_frame_desc().height,
      CUFFT_C2C);
  }

  void PipelineRessources::delete_plan2d()
  {
    cufftDestroy(plan2d_);
  }

  void PipelineRessources::new_gpu_lens()
  {
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));
  }

  void PipelineRessources::delete_gpu_lens()
  {
    cudaFree(gpu_lens_);
  }
}