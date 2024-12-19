/*! \file
 *
 * \brief declaration of texture update function
 */
#pragma once

#include <surface_functions.h>
#include "common.cuh"

void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t stream);
