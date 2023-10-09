/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "composite_struct.hh"

// Computes 3 different p slices and put them in each color
void rgb(cuComplex* input,
         float* output,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         holovibes::RGBWeights weights,
         const cudaStream_t stream);

void postcolor_normalize(float* output,
                         const size_t frame_res,
                         const uint real_line_size,
                         holovibes::units::RectFd selection,
                         const uchar pixel_depth,
                         holovibes::RGBWeights weights,
                         float* averages,
                         const cudaStream_t stream);
