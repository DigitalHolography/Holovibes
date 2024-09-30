/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "composite_struct.hh"

void postcolor_normalize(float* output,
                              const uint fd_height,
                              const uint fd_width,
                              holovibes::units::RectFd selection,
                              float* averages,
                              const cudaStream_t stream);