#include <exception>

#include <cuda_runtime.h>

#include "compute_bundles_2d.hh"
#include "cuda_memory.cuh"
#include "logger.hh"

namespace holovibes
{
UnwrappingResources_2d::UnwrappingResources_2d(const size_t image_size, const cudaStream_t& stream)
    : image_resolution_(image_size)
    , gpu_fx_(nullptr)
    , gpu_fy_(nullptr)
    , gpu_shift_fx_(nullptr)
    , gpu_shift_fy_(nullptr)
    , gpu_angle_(nullptr)
    , gpu_z_(nullptr)
    , gpu_grad_eq_x_(nullptr)
    , gpu_grad_eq_y_(nullptr)
    , minmax_buffer_(nullptr)
    , stream_(stream)
{
    LOG_FUNC(cuda, image_size);

    cudaXMalloc(&gpu_fx_, sizeof(float) * image_resolution_);
    cudaXMalloc(&gpu_fy_, sizeof(float) * image_resolution_);
    cudaXMalloc(&gpu_shift_fx_, sizeof(float) * image_resolution_);
    cudaXMalloc(&gpu_shift_fy_, sizeof(float) * image_resolution_);
    cudaXMalloc(&gpu_angle_, sizeof(float) * image_resolution_);
    cudaXMalloc(&gpu_z_, sizeof(cufftComplex) * image_resolution_);
    cudaXMalloc(&gpu_grad_eq_x_, sizeof(cufftComplex) * image_resolution_);
    cudaXMalloc(&gpu_grad_eq_y_, sizeof(cufftComplex) * image_resolution_);
    cudaXMallocHost(&minmax_buffer_, sizeof(float) * image_resolution_);
}

UnwrappingResources_2d::~UnwrappingResources_2d()
{
    Logger::cuda().trace("UnwrappingResources_2d::~UnwrappingResources_2d()");

    cudaXFree(gpu_fx_);
    cudaXFree(gpu_fy_);
    cudaXFree(gpu_shift_fx_);
    cudaXFree(gpu_shift_fy_);
    cudaXFree(gpu_angle_);
    cudaXFree(gpu_z_);
    cudaXFree(gpu_grad_eq_x_);
    cudaXFree(gpu_grad_eq_y_);
    cudaXFreeHost(minmax_buffer_);
}

void UnwrappingResources_2d::cudaRealloc(void* ptr, const size_t size)
{
    Logger::cuda().trace(" UnwrappingResources_2d::cudaRealloc(size={})", size);

    cudaXFree(ptr);
    cudaXMalloc(&ptr, size);
}

void UnwrappingResources_2d::reallocate(const size_t image_size)
{
    Logger::cuda().trace("UnwrappingResources_2d::reallocate(image_size={})", image_size);

    image_resolution_ = image_size;

    cudaRealloc(gpu_fx_, sizeof(float) * image_resolution_);
    cudaRealloc(gpu_fy_, sizeof(float) * image_resolution_);
    cudaRealloc(gpu_shift_fx_, sizeof(float) * image_resolution_);
    cudaRealloc(gpu_shift_fy_, sizeof(float) * image_resolution_);
    cudaRealloc(gpu_angle_, sizeof(float) * image_resolution_);
    cudaRealloc(gpu_z_, sizeof(cufftComplex) * image_resolution_);
    cudaRealloc(gpu_grad_eq_x_, sizeof(cufftComplex) * image_resolution_);
    cudaRealloc(gpu_grad_eq_y_, sizeof(cufftComplex) * image_resolution_);
    if (minmax_buffer_)
        cudaXFreeHost(minmax_buffer_);
    minmax_buffer_ = nullptr;
    cudaXMallocHost(&minmax_buffer_, sizeof(float) * image_resolution_);
}
} // namespace holovibes
