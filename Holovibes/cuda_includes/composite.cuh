/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

using holovibes::units::RectFd;

void postcolor_normalize(float* output,
                         const uint fd_height,
                         const uint fd_width,
                         RectFd selection,
                         const uchar pixel_depth,
                         float* averages,
                         const cudaStream_t stream);