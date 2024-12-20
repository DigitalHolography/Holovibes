/*! \file
 *
 * \brief Declaration of texture update function
 */
#pragma once

#include <surface_functions.h>
#include "common.cuh"

/*!
 * \brief Update a texture with a frame
 *
 * \param[in] cuSurface The CUDA surface object to update
 * \param[in out] frame The frame to update the texture with
 * \param[in] fd The frame descriptor of the frame
 * \param[in] stream The CUDA stream to use
 */
void textureUpdate(cudaSurfaceObject_t cuSurface,
                   void* frame,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t stream);
