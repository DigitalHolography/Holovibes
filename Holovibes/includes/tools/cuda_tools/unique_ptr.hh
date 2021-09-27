/*! \file
 *
 * \brief std::unique_ptr "specialization" for cudaFree
 */
#pragma once

#include <functional>
#include <memory>

#include <cstddef>

#include "logger.hh"
#include "common.cuh"

namespace holovibes
{
/*! \brief Contains memory handlers for cuda buffers. */
namespace cuda_tools
{
/*! \brief #TODO Add a description for this namespace or remove it */
namespace _private
{
/*! \struct element_size<>
 *
 * \brief #TODO Add a description for this struct
 */
template <typename T>
struct element_size
{
    static const size_t value = sizeof(T);
};

/*! \struct element_size<void>
 *
 * \brief #TODO Add a description for this struct
 */
template <>
struct element_size<void>
{
    static const size_t value = 1;
};
} // namespace _private

/*! \class UniquePtr
 *
 * \brief A smart pointer made for ressources that need to be cudaFreed
 */
template <typename T>
class UniquePtr : public std::unique_ptr<T, std::function<void(T*)>>
{
  public:
    using base = std::unique_ptr<T, std::function<void(T*)>>;
    UniquePtr()
        : base(nullptr, cudaFree)
    {
    }

    UniquePtr(T* ptr)
        : base(ptr, cudaFree)
    {
    }

    /*! \brief Implicit cast operator */
    operator T*() { return get(); }

    /*! \brief Implicit cast operator */
    operator T*() const { return get(); }

    /*! \brief Allocates an array of size sizeof(T) * size */
    UniquePtr(const size_t size)
        : base(nullptr, cudaFree)
    {
        resize(size);
    }

    /*! \brief Allocates an array of size sizeof(T) * size, free the old pointer if not null */
    bool resize(size_t size)
    {
        T* tmp;
        size *= _private::element_size<T>::value;
        reset(nullptr);          // Free itself first
        cudaXMalloc(&tmp, size); // Allocate memory
        reset(tmp);              // Update pointer
        return tmp;
    }
};
} // namespace cuda_tools
} // namespace holovibes
