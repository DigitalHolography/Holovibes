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
      LOG_DEBUG("a");
      return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { 
      LOG_DEBUG("aa");
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

    // CPUUniquePtr()
    //     : val_(nullptr)
    // {
    // }

    // CPUUniquePtr(T* ptr)
    //     : val_(ptr)
    // {
    // }

    T* get() const { 
      LOG_DEBUG("b");
      return val_.get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { 
      LOG_DEBUG("bb");
      return val_.get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    CPUUniquePtr(const size_t size) { resize(size); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        // T* tmp;
        // size *= sizeof(T);
        // tmp = static_cast<T*>(std::realloc(val_.get(), size));
        // LOG_DEBUG("Allocate {:.3f} Gib on Host", static_cast<float>(size) / (1024 * 1024 * 1024));
        // val_.release();
        // val_.reset(tmp);
        // return tmp;

        // T* tmp;
        // size *= sizeof(T);
        // // tmp = static_cast<T*>(std::realloc(val_.get(), size));
        // tmp = static_cast<T*>(std::malloc(size));
        // LOG_DEBUG("Allocate {:.3f} Gib on Host", static_cast<float>(size) / (1024 * 1024 * 1024));
        // val_.reset(nullptr);
        // val_.reset(tmp);
        // return tmp;

        
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
    // std::unique_ptr<T> val_{nullptr};
};

// template <typename T>
// using UniquePtrVariant = std::variant<CudaUniquePtr<T>, CPUUniquePtr<T>>;

/*! \class CudaUniquePtr
 *
 * \brief A smart pointer made for ressources that need to be cudaFreed
 */
template <typename T>
class UniquePtr
{
  public:
    UniquePtr(T* ptr, bool gpu)
        : gpu_(gpu)
    {
      // ptr_ = gpu_ ? CudaUniquePtr<T>(ptr) : CPUUniquePtr<T>(ptr);
      if (gpu_)
        ptr_ = CudaUniquePtr<T>(ptr);
      else 
        ptr_ = CPUUniquePtr<T>(ptr);
    }

    UniquePtr(bool gpu=true)
      : gpu_(gpu)
    {
      // UniquePtrVariant<T> ptr_;
      // ptr_ = CudaUniquePtr<T>();
      // std::variant<CudaUniquePtr<T>, CPUUniquePtr<T>> ptr_;
      // auto tmp = CudaUniquePtr<T>();
      // ptr_.emplace<0>(CudaUniquePtr<T>());

      // std::visit([this](auto&& value) {
      //   using T1 = std::decay_t<decltype(value)>;
      //   ptr_.emplace<T1>(value);
      // }, CudaUniquePtr<T>());
      // if (gpu_)
      //   ptr_.emplace<0>(CudaUniquePtr<T>());
      // else
      //   ptr_.emplace<1>(CPUUniquePtr<T>());
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

    T* get() { 
      return gpu_ ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get();
    }

    /*! \brief Implicit cast operator */
    operator T*() const{
      return gpu_ ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get();
      // return gpu_ ? std::get<0>(ptr_)() : std::get<1>(ptr_)();
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

    //  ~UniquePtr(){};

  private:
    bool gpu_;
    // UniquePtrVariant ptr_;
    std::variant<CudaUniquePtr<T>, CPUUniquePtr<T>> ptr_;

};


} // namespace holovibes::cuda_tools