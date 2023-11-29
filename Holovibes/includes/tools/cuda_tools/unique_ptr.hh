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

/*! \class CudaUniquePtr
 *
 * \brief A smart pointer made for ressources that need to be cudaFreed
 */
template <typename T>
class UniquePtr
{
  public:
    virtual T* get() const = 0;

    /*! \brief Implicit cast operator */
    virtual operator T*() const = 0;

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    virtual bool resize(size_t size) = 0;

    virtual void reset(T* ptr) = 0;

    virtual void reset() = 0;

    virtual ~UniquePtr(){};
};


template <typename T>
class CPUUniquePtr : public UniquePtr<T>
{
  public:
    CPUUniquePtr()
        : val_(nullptr)
    {
    }

    CPUUniquePtr(T* ptr)
        : val_(ptr)
    {
    }

    T* get() const { return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { return val_.get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    CPUUniquePtr(const size_t size) { resize(size); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= sizeof(T);
        tmp = static_cast<T*>(std::realloc(val_.get(), size));
        LOG_DEBUG("Allocate {:.3f} Gib on Host", static_cast<float>(size) / (1024 * 1024 * 1024));
        val_.release();
        val_.reset(tmp);
        return tmp;
    }

    void reset(T* ptr) { return val_.reset(ptr); }
    void reset() { return val_.reset(nullptr); }

  protected:
    std::unique_ptr<T> val_{nullptr};
};


template <typename T>
class CudaUniquePtr : public UniquePtr<T>
{
  public:
    CudaUniquePtr()
        : val_(nullptr, cudaXFree)
    {
    }

    CudaUniquePtr(T* ptr)
        : val_(ptr, cudaXFree)
    {
    }

    T* get() const { return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { return val_.get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    CudaUniquePtr(const size_t size) { resize(size); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= sizeof(T);
        val_.reset(nullptr);     // Free itself first
        cudaXMalloc(&tmp, size); // Allocate memory
        val_.reset(tmp);         // Update pointer
        return tmp;
    }

    void reset(T* ptr) { return val_.reset(ptr); }
    void reset() { return val_.reset(nullptr); }

  protected:
    std::unique_ptr<T, decltype(cudaXFree)*> val_{nullptr, cudaXFree};
};
} // namespace holovibes::cuda_tools
