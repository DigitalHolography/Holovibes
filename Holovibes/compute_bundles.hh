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
    /*! Initialize the capacity from Global configuration,
     * set size and next_index to zero, and buffers pointers to null pointers. */
    UnwrappingResources();

    /*! If buffers were allocated, deallocate them. */
    ~UnwrappingResources();

    /*! Allocates all buffers based on the new image size. Forces reallocation.
    *
    * \param image_size The number of pixels in an image. */
    void allocate(const size_t image_size);

    size_t capacity_; //!< Maximum number of matrices kept in history.
    size_t size_; //!< Current number of matrices kept in history.
    unsigned next_index_; //!< Index of the next matrix to be overriden (the oldest).
    /*! Buffer used to cumulate phase adjustments, before they can be
    * applied back in phase unwrapping. Phase being an angle, it is one
    * part of a complex information, and can be stored in a float. */
    float* gpu_unwrap_buffer_;

    cufftComplex* gpu_predecessor_;
    cufftComplex* gpu_diff_predecessor_;
    cufftComplex* gpu_diff_;
    cufftComplex* gpu_global_diff_;

    /*! Copy of the previous frame's angle values. Updated over unwrapping. */
    float* gpu_angle_predecessor_;
    /*! Copy of the current frame's angle values. Used locally in unwrapping. */
    float* gpu_angle_current_;
  };
}