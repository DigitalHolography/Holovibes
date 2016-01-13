#include "compute_bundles.hh"

namespace holovibes
{
  UnwrappingResources::UnwrappingResources(
    const unsigned capacity,
    const size_t image_size)
    : total_memory_(capacity)
    , capacity_(capacity)
    , size_(0)
    , next_index_(0)
    , gpu_unwrap_buffer_(nullptr)
    , gpu_predecessor_(nullptr)
    , gpu_angle_predecessor_(nullptr)
    , gpu_angle_current_(nullptr)
    , gpu_angle_copy_(nullptr)
    , gpu_unwrapped_angle_(nullptr)
  {
    auto nb_unwrap_elts = image_size * capacity_;

    cudaMalloc(&gpu_unwrap_buffer_, sizeof(float)* nb_unwrap_elts);
    /* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
    cudaMemset(gpu_unwrap_buffer_, 0, sizeof(float)* nb_unwrap_elts);

    cudaMalloc(&gpu_predecessor_, sizeof(cufftComplex)* image_size);

    cudaMalloc(&gpu_angle_predecessor_, sizeof(float)* image_size);

    cudaMalloc(&gpu_angle_current_, sizeof(float)* image_size);

    cudaMalloc(&gpu_angle_copy_, sizeof(float)* image_size);

    cudaMalloc(&gpu_unwrapped_angle_, sizeof(float)* image_size);
  }

  UnwrappingResources::~UnwrappingResources()
  {
    if (gpu_unwrap_buffer_)
      cudaFree(gpu_unwrap_buffer_);
    if (gpu_predecessor_)
      cudaFree(gpu_predecessor_);
    if (gpu_angle_predecessor_)
      cudaFree(gpu_angle_predecessor_);
    if (gpu_angle_current_)
      cudaFree(gpu_angle_current_);
    if (gpu_angle_copy_)
      cudaFree(gpu_angle_copy_);
    if (gpu_unwrapped_angle_)
      cudaFree(gpu_unwrapped_angle_);
  }

  void UnwrappingResources::reallocate(const size_t image_size)
  {
    // We compare requested memory against available memory, and reallocate if needed.
    if (capacity_ <= total_memory_)
      return;

    total_memory_ = capacity_;
    auto nb_unwrap_elts = image_size * capacity_;

    if (gpu_unwrap_buffer_)
      cudaFree(gpu_unwrap_buffer_);
    cudaMalloc(&gpu_unwrap_buffer_, sizeof(float)* nb_unwrap_elts);
    /* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
    cudaMemset(gpu_unwrap_buffer_, 0, sizeof(float)* nb_unwrap_elts);

    if (gpu_predecessor_)
      cudaFree(gpu_predecessor_);
    cudaMalloc(&gpu_predecessor_, sizeof(cufftComplex)* image_size);

    if (gpu_angle_predecessor_)
      cudaFree(gpu_angle_predecessor_);
    cudaMalloc(&gpu_angle_predecessor_, sizeof(float)* image_size);

    if (gpu_angle_current_)
      cudaFree(gpu_angle_current_);
    cudaMalloc(&gpu_angle_current_, sizeof(float)* image_size);

    if (gpu_angle_copy_)
      cudaFree(gpu_angle_copy_);
    cudaMalloc(&gpu_angle_copy_, sizeof(float)* image_size);

    if (gpu_unwrapped_angle_)
      cudaFree(gpu_unwrapped_angle_);
    cudaMalloc(&gpu_unwrapped_angle_, sizeof(float)* image_size);
  }

  void UnwrappingResources::reset(const size_t capacity)
  {
    capacity_ = capacity;
    size_ = 0;
    next_index_ = 0;
  }
}