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
         bool normalize,
         const ushort red,
         const ushort blue,
         holovibes::RGBWeights weights,
         const cudaStream_t stream);

void postcolor_normalize(float* output,
                         const size_t frame_res,
                         const uint real_line_size,
                         holovibes::units::RectFd selection,
                         holovibes::RGBWeights weights,
                         const cudaStream_t stream);
