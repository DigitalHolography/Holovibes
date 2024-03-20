/*! \file
 *
 * \brief std::unique_ptr "specialization" for cudaFree
 */
#pragma once

#include <functional>
#include <memory>
#include <variant>

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
class CudaUniquePtr
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

      
    T* get() const { 
      // LOG_DEBUG("a");
      return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { 
      // LOG_DEBUG("aa");
      return val_.get(); }

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

/*! \class CPUUniquePtr
 *
 * \brief A smart wrapper around unique_ptr, used as a CPU alternative of CudaUniquePtr
 */
template <typename T>
class CPUUniquePtr
{
  public:
    CPUUniquePtr()
        : val_(nullptr, cudaXFreeHost)
    {
    }

    CPUUniquePtr(T* ptr)
        : val_(ptr, cudaXFreeHost)
    {
    }


    T* get() const { 
      return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { 
      return val_.get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    CPUUniquePtr(const size_t size) { resize(size); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= sizeof(T);
        val_.reset(nullptr);     // Free itself first
        cudaXMallocHost(&tmp, size); // Allocate memory
        val_.reset(tmp);         // Update pointer
        return tmp;
    }

    void reset(T* ptr) { return val_.reset(ptr); }
    void reset() { return val_.reset(nullptr); }

  protected:
    std::unique_ptr<T, decltype(cudaXFreeHost)*> val_{nullptr, cudaXFreeHost};
};

/*! \class UniquePtr
 *
 * \brief A wrapper around a variant of CPU/CudaUniquePtr
 */
template <typename T>
class UniquePtr
{
  public:
    UniquePtr(T* ptr, bool gpu)
        : gpu_(gpu)
    {
      if (gpu_)
        ptr_ = CudaUniquePtr<T>(ptr);
      else 
        ptr_ = CPUUniquePtr<T>(ptr);
    }

    UniquePtr(bool gpu=true)
      : gpu_(gpu)
    {
      if (gpu_)
        ptr_ = CudaUniquePtr<T>();
      else 
        ptr_ = CPUUniquePtr<T>();
    }

    UniquePtr(T* ptr)
    {
      UniquePtr(ptr, true);
    }

    UniquePtr(const size_t size) { 
      resize(size); 
    }

    T* get() const { 
      return gpu_ ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get();
    }

    /*! \brief Implicit cast operator */
    operator T*() const{
      return gpu_ ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get();
    }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size){
      return gpu_ ? std::get<0>(ptr_).resize(size) : std::get<1>(ptr_).resize(size);
    }

     void reset(T* ptr){
      return gpu_ ? std::get<0>(ptr_).reset(ptr) : std::get<1>(ptr_).reset(ptr);
     }

     void reset(){
      return gpu_ ? std::get<0>(ptr_).reset() : std::get<1>(ptr_).reset();
     }

  private:
    bool gpu_;
    std::variant<CudaUniquePtr<T>, CPUUniquePtr<T>> ptr_;

};
} // namespace holovibes::cuda_tools