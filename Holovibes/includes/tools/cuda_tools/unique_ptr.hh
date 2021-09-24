/*! \file
 *
 * \brief std::unique_ptr "specialization" for cudaFree
 */
#pragma once

#include <functional>
#include <memory>

#include <cstddef>

#include "cuda_memory.cuh"
#include "logger.hh"
#include "common.cuh"

/*! \brief Contains memory handlers for cuda buffers. */
namespace holovibes::cuda_tools
{

/*! \class UniquePtr
 *
 * \brief A smart pointer made for ressources that need to be cudaFreed
 */
template <typename T>
class UniquePtr : public std::unique_ptr<T, decltype(cudaXFree)*>
{
  public:
    using base = std::unique_ptr<T, decltype(cudaXFree)*>;
    UniquePtr()
        : base(nullptr, cudaXFree)
    {
    }

    UniquePtr(T* ptr)
        : base(ptr, cudaXFree)
    {
    }

    /*! \brief Implicit cast operator */
    operator T*() { return get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { return get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    UniquePtr(const size_t size)
        : base(nullptr, cudaXFree)
    {
        resize(size);
    }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= sizeof(T);
        reset(nullptr);          // Free itself first
        cudaXMalloc(&tmp, size); // Allocate memory
        reset(tmp);              // Update pointer
        return tmp;
    }
};
} // namespace holovibes::cuda_tools
