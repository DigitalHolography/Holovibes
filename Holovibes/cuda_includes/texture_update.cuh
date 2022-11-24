/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <surface_functions.h>
#include "common.cuh"

void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const holovibes::FrameDescriptor& fd,
                   const cudaStream_t stream);
