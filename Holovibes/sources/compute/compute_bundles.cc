/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <exception>

#include <cuda_runtime.h>

#include "compute_bundles.hh"
#include "cuda_memory.cuh"

namespace holovibes
{
UnwrappingResources::UnwrappingResources(const unsigned capacity,
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

    cudaXMalloc(&gpu_unwrap_buffer_, sizeof(float) * nb_unwrap_elts);
    cudaXMalloc(&gpu_predecessor_, sizeof(cufftComplex) * image_size);
    cudaXMalloc(&gpu_angle_predecessor_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_angle_current_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_angle_copy_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_unwrapped_angle_, sizeof(float) * image_size);

    /* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
    cudaXMemset(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts);
}

UnwrappingResources::~UnwrappingResources()
{
    cudaXFree(gpu_unwrap_buffer_);
    cudaXFree(gpu_predecessor_);
    cudaXFree(gpu_angle_predecessor_);
    cudaXFree(gpu_angle_current_);
    cudaXFree(gpu_angle_copy_);
    cudaXFree(gpu_unwrapped_angle_);
}

void UnwrappingResources::cudaRealloc(void* ptr, const size_t size)
{
    cudaXFree(ptr);
    cudaXMalloc(&ptr, size);
}

void UnwrappingResources::reallocate(const size_t image_size)
{
    // We compare requested memory against available memory, and reallocate if
    // needed.
    if (capacity_ <= total_memory_)
        return;

    total_memory_ = capacity_;
    auto nb_unwrap_elts = image_size * capacity_;

    cudaRealloc(gpu_unwrap_buffer_, sizeof(float) * nb_unwrap_elts);
    cudaRealloc(gpu_predecessor_, sizeof(cufftComplex) * image_size);
    cudaRealloc(gpu_angle_predecessor_, sizeof(float) * image_size);
    cudaRealloc(gpu_angle_current_, sizeof(float) * image_size);
    cudaRealloc(gpu_angle_copy_, sizeof(float) * image_size);
    cudaRealloc(gpu_unwrapped_angle_, sizeof(float) * image_size);
    cudaXMemset(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts);
}

void UnwrappingResources::reset(const size_t capacity)
{
    capacity_ = capacity;
    size_ = 0;
    next_index_ = 0;
}
} // namespace holovibes