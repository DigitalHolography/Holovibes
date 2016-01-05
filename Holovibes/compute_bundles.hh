#pragma once

# include <cuda_runtime.h>

# include "config.hh"

namespace holovibes
{
  /*! Regroup all resources used for phase unwrapping.
  * Takes care of initialization and destruction. */
  struct UnwrappingResources
  {
    UnwrappingResources();

    ~UnwrappingResources();

    void allocate(const size_t image_size);

    size_t capacity_;
    size_t size_;
    unsigned next_index_;
    /*! Buffer used to cumulate phase adjustments, before they can be
    * applied back in phase unwrapping. Phase being an angle, it is one
    * part of a complex information, and can be stored in a float. */
    float* gpu_unwrap_buffer_;

    /*! Copy of the previous frame's angle values. Updated over unwrapping. */
    float* gpu_angle_predecessor_;
    /*! Copy of the current frame's angle values. Used locally in unwrapping. */
    float* gpu_angle_current_;
  };
}