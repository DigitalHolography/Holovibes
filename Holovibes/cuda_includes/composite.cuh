/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

// Computes 3 different p slices and put them in each color
void rgb(cuComplex* input,
         float* output,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         const float weight_r,
         const float weight_g,
         const float weight_b,
         const cudaStream_t stream);

void postcolor_normalize(float* output,
                         const size_t frame_res,
                         const uint real_line_size,
                         holovibes::units::RectFd selection,
                         const uchar pixel_depth,
                         const float weight_r,
                         const float weight_g,
                         const float weight_b,
                         float* averages,
                         const cudaStream_t stream);
