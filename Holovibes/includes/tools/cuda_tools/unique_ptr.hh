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
class UniquePtr
{
  public:
    UniquePtr()
        : val_(nullptr, cudaXFree)
    {
    }

    UniquePtr(T* ptr)
        : val_(ptr, cudaXFree)
    {
    }

    T* get() const {
        return val_.get();
    }

    /*! \brief Implicit cast operator */
    operator T*() const { return &(*val_); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    UniquePtr(const size_t size)
    {
        resize(size);
    }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= sizeof(T);
        val_.reset(nullptr);          // Free itself first
        cudaXMalloc(&tmp, size);      // Allocate memory
        val_.reset(tmp);              // Update pointer
        return tmp;
    }

    void reset(T* ptr) { return val_.reset(ptr); }
    void reset() { return val_.reset(nullptr); }

protected:
    std::unique_ptr<T, decltype(cudaXFree)*> val_{nullptr, cudaXFree};
};
} // namespace holovibes::cuda_tools
