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
#include "enum_device.hh"

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

    T* get() const
    {
        // LOG_DEBUG("a");
        return val_.get();
    }

    /*! \brief Implicit cast operator */
    operator T*() const
    {
        // LOG_DEBUG("aa");
        return val_.get();
    }

    /*! \brief Allocates an array of size sizeof(T) * size */
    CudaUniquePtr(const size_t size) { resize(size); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null
     * \param size The size of the array to allocate
     * \return true if the allocation was successful, false otherwise
     */
    bool resize(size_t size)
    {
        val_.reset(nullptr); // Free itself first

        T* tmp = nullptr;
        size *= sizeof(T);
        size_ = size;
        cudaXMalloc(&tmp, size); // Allocate memory
        val_.reset(tmp);         // Update pointer
        return tmp != nullptr;
    }

    size_t get_size() const { return size_; }

    void reset(T* ptr) { val_.reset(ptr); }
    void reset() { val_.reset(nullptr); }

  protected:
    std::unique_ptr<T, decltype(cudaXFree)*> val_{nullptr, cudaXFree};
    size_t size_;
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
        size_ = size;
        val_.reset(nullptr);         // Free itself first
        cudaXMallocHost(&tmp, size); // Allocate memory
        val_.reset(tmp);             // Update pointer
        return tmp;
    }

    size_t get_size() const { return size_; }

    void reset(T* ptr) { return val_.reset(ptr); }
    void reset() { return val_.reset(nullptr); }

  protected:
    std::unique_ptr<T, decltype(cudaXFreeHost)*> val_{nullptr, cudaXFreeHost};
    size_t size_;
};

/*! \class UniquePtr
 *
 * \brief A wrapper around a variant of CPU/CudaUniquePtr
 */
template <typename T>
class UniquePtr
{
  public:
    UniquePtr(T* ptr, Device device)
        : device_(device)
    {
        if (device_ == Device::GPU)
            ptr_ = CudaUniquePtr<T>(ptr);
        else
            ptr_ = CPUUniquePtr<T>(ptr);
    }

    UniquePtr(Device device = Device::GPU)
        : device_(device)
    {
        if (device_ == Device::GPU)
            ptr_ = CudaUniquePtr<T>();
        else
            ptr_ = CPUUniquePtr<T>();
    }

    UniquePtr(T* ptr) { UniquePtr(ptr, Device::GPU); }

    UniquePtr(const size_t size) { resize(size); }

    T* get() const { return device_ == Device::GPU ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get(); }

    size_t get_size() const
    {
        return device_ == Device::GPU ? std::get<0>(ptr_).get_size() : std::get<1>(ptr_).get_size();
    }

    /*! \brief Implicit cast operator */
    operator T*() const { return device_ == Device::GPU ? std::get<0>(ptr_).get() : std::get<1>(ptr_).get(); }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        return device_ == Device::GPU ? std::get<0>(ptr_).resize(size) : std::get<1>(ptr_).resize(size);
    }

    void reset(T* ptr) { return device_ == Device::GPU ? std::get<0>(ptr_).reset(ptr) : std::get<1>(ptr_).reset(ptr); }

    void reset() { return device_ == Device::GPU ? std::get<0>(ptr_).reset() : std::get<1>(ptr_).reset(); }

  private:
    Device device_;
    std::variant<CudaUniquePtr<T>, CPUUniquePtr<T>> ptr_;
};
} // namespace holovibes::cuda_tools
