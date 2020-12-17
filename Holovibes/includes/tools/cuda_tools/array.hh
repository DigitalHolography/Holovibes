/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * cuda_tools::UniquePtr containing an array */
#pragma once

#include <cuda_runtime_api.h>
#include <fstream>

#include "cuda_memory.cuh"

namespace holovibes
{
namespace cuda_tools
{
/// Array class for cuda buffers that ocasionally need to be resized
template <typename T>
class Array : public UniquePtr<T>
{
  public:
    using base = UniquePtr<T>;

    /// Intantiate an empty / nullptr array
    Array()
        : base()
        , size_(0)
    {
    }

    /// Creates an array of size sizeof(T) * size
    Array(std::size_t size)
        : base(size)
        , size_(size)
    {
    }

    /// Realloc the array only if needed
    ///
    /// \return if the resize succeeded
    bool ensure_minimum_size(std::size_t size)
    {
        if (size <= size_)
            return true;
        resize(size);
        if (get())
        {
            size_ = size;
            return true;
        }
        size_ = 0;
        return false;
    }

    /// Is the array size greater or equal to size
    bool is_large_enough(std::size_t size) const { return size_ >= size; }

    /// Resize the array
    void resize(std::size_t size)
    {
        base::resize(size);
        size_ = size;
    }

    /// Override reset to set the size accordingly
    void reset(T* ptr = nullptr)
    {
        base::reset(ptr);
        size_ = 0;
    }

    /// Dumps all the array into a file
    ///
    /// Slow and inefficient, for debug purpose only
    void write_to_file(std::string filename, bool trunc = false)
    {
        auto cpu_buffer = to_cpu();
        const uint byte_size = size_ * sizeof(T);
        std::ofstream file(filename,
                           std::ios::binary |
                               (trunc ? std::ios::trunc : std::ios::app));
        file.write(reinterpret_cast<char*>(cpu_buffer.data()), byte_size);
    }

    /// Dumps all the array into a CPU vector
    ///
    /// Slow and inefficient, for debug purpose only
    std::vector<T> to_cpu()
    {
        std::vector<T> cpu_buffer(size_);
        const size_t byte_size = size_ * sizeof(T);
        cudaXMemcpy(cpu_buffer.data(),
                    get(),
                    byte_size,
                    cudaMemcpyDeviceToHost);
        return cpu_buffer;
    }

  private:
    std::size_t size_;
};
} // namespace cuda_tools
} // namespace holovibes