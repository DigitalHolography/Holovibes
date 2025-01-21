#include <exception>

#include <cuda_runtime.h>

#include "compute_bundles.hh"
#include "cuda_memory.cuh"

#include "logger.hh"

namespace holovibes
{
UnwrappingResources::UnwrappingResources(const unsigned capacity, const size_t image_size, const cudaStream_t& stream)
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
    , stream_(stream)
{
    LOG_FUNC(capacity, image_size);

    auto nb_unwrap_elts = image_size * capacity_;

    cudaXMalloc(&gpu_unwrap_buffer_, sizeof(float) * nb_unwrap_elts);
    cudaXMalloc(&gpu_predecessor_, sizeof(cufftComplex) * image_size);
    cudaXMalloc(&gpu_angle_predecessor_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_angle_current_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_angle_copy_, sizeof(float) * image_size);
    cudaXMalloc(&gpu_unwrapped_angle_, sizeof(float) * image_size);

    /* Cumulative phase adjustments in gpu_unwrap_buffer are reset. */
    cudaXMemsetAsync(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts, stream_);
}

UnwrappingResources::~UnwrappingResources()
{
    LOG_FUNC();

    cudaXFree(gpu_unwrap_buffer_);
    cudaXFree(gpu_predecessor_);
    cudaXFree(gpu_angle_predecessor_);
    cudaXFree(gpu_angle_current_);
    cudaXFree(gpu_angle_copy_);
    cudaXFree(gpu_unwrapped_angle_);
}

void UnwrappingResources::cudaRealloc(void* ptr, const size_t size)
{
    LOG_FUNC(size);

    cudaXFree(ptr);
    cudaXMalloc(&ptr, size);
}

void UnwrappingResources::reallocate(const size_t image_size)
{
    LOG_FUNC(image_size);

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
    cudaXMemsetAsync(gpu_unwrap_buffer_, 0, sizeof(float) * nb_unwrap_elts, stream_);
}

void UnwrappingResources::reset(const size_t capacity)
{
    LOG_FUNC(capacity);

    capacity_ = capacity;
    size_ = 0;
    next_index_ = 0;
}
} // namespace holovibes
