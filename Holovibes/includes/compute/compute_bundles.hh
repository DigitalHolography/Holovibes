/*! \file
 *
 * \brief Regroup all resources used for phase unwrapping.
 */
#pragma once

#include <cufft.h>

namespace holovibes
{
/*! \struct UnwrappingResources
 *
 * \brief  Regroup all resources used for phase unwrapping.
 *
 * Takes care of initialization and destruction.
 */
struct UnwrappingResources
{
    /*! \brief Allocate with CUDA the required memory, and initialize the needed variables */
    UnwrappingResources(const unsigned capacity, const size_t image_size, const cudaStream_t& stream);

    /*! \brief If buffers were allocated, deallocate them. */
    ~UnwrappingResources();

    /*! \brief Allocates all buffers based on the new image size.
     *
     * Reallocation is carried out only if the total amount of images
     * that can be stored in gpu_unwrap_buffer_ is inferior to
     * the capacity requested (in capacity_).
     *
     * \param image_size The number of pixels in an image.
     */
    void cudaRealloc(void* ptr, const size_t size);
    void reallocate(const size_t image_size);

    /*! \brief Update history size without causing reallocation.
     *
     * If you wish to rearrange memory, you should call reallocate().
     * This function simply changes the capacity and resets the size
     * of the buffer to zero, effectively losing track of previous data.
     */
    void reset(const size_t capacity);

    /*! \brief The real number of matrices reserved in memory.
     *
     * Not all may be used. The purpose is to avoid requesting too much memory
     * operations to the graphics card, which causes crashes.
     */
    size_t total_memory_;
    /*! \brief Maximum number of matrices kept in history. */
    size_t capacity_;
    /*! \brief Current number of matrices kept in history. */
    size_t size_;

    /*! \brief Index of the next matrix to be overriden (the oldest). */
    unsigned next_index_;
    /*! \brief Buffer used to cumulate phase images.
     *
     * Phase being an angle, it is one part of a complex information, and can be stored in a float.
     * Phase images stored here are summed up together at each iteration and added to the latest phase image.
     */
    float* gpu_unwrap_buffer_;

    /*! \brief Copy of the previous complex (untouched) image. */
    cufftComplex* gpu_predecessor_;
    /*! \brief Copy of the previous frame's angle values. Updated over unwrapping. */
    float* gpu_angle_predecessor_;
    /*! \brief Copy of the current frame's angle values. Target of the gpu_unwrap_buffer summed elements. */
    float* gpu_angle_current_;
    /*! \brief Copy of the current frame's angle values, before it is summed to gpu_unwrap_buffer's elements. */
    float* gpu_angle_copy_;
    /*! \brief Current phase image after unwrapping. */
    float* gpu_unwrapped_angle_;

    const cudaStream_t& stream_;
};

} // namespace holovibes
