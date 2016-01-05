#include "compute_bundles.hh"

namespace holovibes
{
  UnwrappingResources::UnwrappingResources()
    : capacity_(Global::global_config.unwrap_history_size)
    , size_(0)
    , next_index_(0)
    , gpu_unwrap_buffer_(nullptr)
    , gpu_angle_predecessor_(nullptr)
    , gpu_angle_current_(nullptr)
  {
  }

  UnwrappingResources::~UnwrappingResources()
  {
    if (gpu_unwrap_buffer_)
      cudaFree(gpu_unwrap_buffer_);
    if (gpu_angle_predecessor_)
      cudaFree(gpu_angle_predecessor_);
    if (gpu_angle_current_)
      cudaFree(gpu_angle_current_);
  }

  void UnwrappingResources::allocate(const size_t image_size)
  {
    /* Phase unwrapping requires a reference. We shall copy the first frame
    * obtained into gpu_angle_predecessor_, for initialization. This is done in
    * the unwrap function. The first iteration will have no effect, because
    * the frame will be compared to itself.
    * Also, cumulative phase adjustments in gpu_unwrap_buffer are reset. */
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
  }
}