#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# include "config.hh"

namespace holovibes
{
  /*! Regroup all resources used for phase unwrapping.
  * Takes care of initialization and destruction. */
  struct UnwrappingResources
  {
    /*! Initialize the capacity from history_size,
     * set size and next_index to zero, and buffers pointers to null pointers. */
    UnwrappingResources(const unsigned capacity, const size_t image_size);

    /*! If buffers were allocated, deallocate them. */
    ~UnwrappingResources();

    /*! Allocates all buffers based on the new image size.
     *
     * Reallocation is carried out only if the total amount of images
     * that can be stored in gpu_unwrap_buffer_ is inferior to
     * the capacity requested (in capacity_).
     *
     * \param image_size The number of pixels in an image. */
    void reallocate(const size_t image_size);

    /*! Update history size without causing reallocation.
     *
     * If you wish to rearrange memory, you should call reallocate().
     * This function simply changes the capacity and resets the size
     * of the buffer to zero, effectively losing track of previous data. */
    void reset(const size_t capacity);

    /*! The real number of matrices reserved in memory.
     * Not all may be used. The purpose is to avoid requesting too much memory
     * operations to the graphics card, which causes crashes. */
    size_t total_memory_;
    size_t capacity_; //!< Maximum number of matrices kept in history.
    size_t size_; //!< Current number of matrices kept in history.

    unsigned next_index_; //!< Index of the next matrix to be overriden (the oldest).
    /*! Buffer used to cumulate phase images. Phase being an angle, it is one
    * part of a complex information, and can be stored in a float.
    * Phase images stored here are summed up together at each iteration and
    * added to the latest phase image. */
    float* gpu_unwrap_buffer_;

    /*! Copy of the previous complex (untouched) image. */
    cufftComplex* gpu_predecessor_;
    /*! Copy of the previous frame's angle values. Updated over unwrapping. */
    float* gpu_angle_predecessor_;
    /*! Copy of the current frame's angle values.
     * Target of the gpu_unwrap_buffer summed elements. */
    float* gpu_angle_current_;
    /*! Copy of the current frame's angle values, before it is summed
     * to gpu_unwrap_buffer's elements. */
    float* gpu_angle_copy_;
    /*! Current phase image after unwrapping. */
    float* gpu_unwrapped_angle_;
  };
}