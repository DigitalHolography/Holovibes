#include "compute_bundles.hh"

namespace holovibes
{
  UnwrappingResources::UnwrappingResources(const unsigned capacity)
    : capacity_(capacity)
    , size_(0)
    , next_index_(0)
    , gpu_unwrap_buffer_(nullptr)
    , gpu_predecessor_(nullptr)
    , gpu_diff_(nullptr)
    , gpu_angle_predecessor_(nullptr)
    , gpu_angle_current_(nullptr)
  {
  }

  UnwrappingResources::~UnwrappingResources()
  {
    if (gpu_unwrap_buffer_)
      cudaFree(gpu_unwrap_buffer_);
    if (gpu_predecessor_)
      cudaFree(gpu_predecessor_);
    if (gpu_diff_)
      cudaFree(gpu_diff_);
    if (gpu_angle_predecessor_)
      cudaFree(gpu_angle_predecessor_);
    if (gpu_angle_current_)
      cudaFree(gpu_angle_current_);
  }

  void UnwrappingResources::allocate(const size_t image_size)
  {
    /* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
    auto nb_unwrap_elts = image_size * capacity_;
    if (gpu_unwrap_buffer_)
      cudaFree(gpu_unwrap_buffer_);
    cudaMalloc(&gpu_unwrap_buffer_, sizeof(float)* nb_unwrap_elts);
    cudaMemset(gpu_unwrap_buffer_, 0, sizeof(float)* nb_unwrap_elts);

    if (gpu_angle_predecessor_)
      cudaFree(gpu_angle_predecessor_);
    cudaMalloc(&gpu_angle_predecessor_, sizeof(float)* image_size);

    if (gpu_angle_current_)
      cudaFree(gpu_angle_current_);
    cudaMalloc(&gpu_angle_current_, sizeof(float)* image_size);

    if (gpu_predecessor_)
      cudaFree(gpu_predecessor_);
    cudaMalloc(&gpu_predecessor_, sizeof(cufftComplex)* image_size);

    if (gpu_diff_)
      cudaFree(gpu_diff_);
    cudaMalloc(&gpu_diff_, sizeof(cufftComplex)* image_size);
  }

  void UnwrappingResources::change_capacity(const size_t capacity)
  {
    capacity_ = capacity;
  }
}