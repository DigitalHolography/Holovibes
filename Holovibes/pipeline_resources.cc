#include "stdafx.h"

#include <cuda_runtime.h>
#include <cassert>

#include "pipeline_resources.hh"
#include "preprocessing.cuh"

namespace holovibes
{
  PipelineResources::PipelineResources(
    Queue& input,
    Queue& output,
    unsigned short n)
    : input_(input)
    , output_(output)
    , gpu_sqrt_vector_(nullptr)
    , gpu_output_buffer_(nullptr)
    , plan3d_(0)
    , plan2d_(0)
    , gpu_lens_(nullptr)
    , gpu_input_buffer_(nullptr)
    , gpu_output_frame_(nullptr)
  {
    assert(n != 0 && "n parameter can not be 0");
    new_gpu_sqrt_vector(sqrt_vector_size);
    new_gpu_output_buffer(n);
    new_plan3d(n);
    new_plan2d();
    new_gpu_lens();
    new_gpu_input_buffer(n);
  }

  PipelineResources::~PipelineResources()
  {
    delete_gpu_sqrt_vector();
    delete_gpu_output_buffer();
    delete_plan3d();
    delete_plan2d();
    delete_gpu_lens();
    delete_gpu_input_buffer();
  }

  /* Public methods */

  void PipelineResources::update_n_parameter(unsigned short n)
  {
    assert(n != 0 && "n must be strictly positive");
    delete_gpu_output_buffer();
    new_gpu_output_buffer(n);
    delete_plan3d();
    new_plan3d(n);
    delete_gpu_input_buffer();
    new_gpu_input_buffer(n);
  }

  /* Private methods */

  void PipelineResources::new_gpu_sqrt_vector(unsigned short n)
  {
    cudaMalloc<float>(&gpu_sqrt_vector_, sizeof(float) * n);
    make_sqrt_vect(gpu_sqrt_vector_, n);
  }

  void PipelineResources::delete_gpu_sqrt_vector()
  {
    cudaFree(gpu_sqrt_vector_);
  }

  void PipelineResources::new_gpu_output_buffer(unsigned short n)
  {
    cudaMalloc<unsigned short>(&gpu_output_buffer_,
      sizeof(unsigned short) * input_.get_pixels() * n);
  }

  void PipelineResources::delete_gpu_output_buffer()
  {
    cudaFree(gpu_output_buffer_);
  }

  void PipelineResources::new_plan3d(unsigned short n)
  {
    cufftPlan3d(
      &plan3d_,
      n,                              // NX
      input_.get_frame_desc().width,  // NY
      input_.get_frame_desc().height, // NZ
      CUFFT_C2C);
  }

  void PipelineResources::delete_plan3d()
  {
    cufftDestroy(plan3d_);
  }

  void PipelineResources::new_plan2d()
  {
    cufftPlan2d(
      &plan2d_,
      input_.get_frame_desc().width,
      input_.get_frame_desc().height,
      CUFFT_C2C);
  }

  void PipelineResources::delete_plan2d()
  {
    cufftDestroy(plan2d_);
  }

  void PipelineResources::new_gpu_lens()
  {
    cudaMalloc(&gpu_lens_,
      input_.get_pixels() * sizeof(cufftComplex));
  }

  void PipelineResources::delete_gpu_lens()
  {
    cudaFree(gpu_lens_);
  }

  void PipelineResources::new_gpu_input_buffer(unsigned short n)
  {
    cudaMalloc<cufftComplex>(&gpu_input_buffer_,
      sizeof(cufftComplex) * input_.get_pixels() * n);
  }

  void PipelineResources::delete_gpu_input_buffer()
  {
    cudaFree(gpu_input_buffer_);
  }
}