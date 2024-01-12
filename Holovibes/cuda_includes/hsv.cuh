/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"
typedef unsigned int uint;

enum HSV {
    H = 0,
    S = 1,
    V = 2
};


// The operation used after the channel values computation, used to renormalize the values
enum threshold_op
{
    CLAMP, // Clamp values under min_val to min_val and values over max_val to max_val
    CRUSH, // Crush the gradient from min_val to max_val
    ZOOM  // Rescale values from [0,1] to [min_val, max_val]
};

void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct);

/*
void hsv_cuts(const float* gpu_in_cut_xz,
              const float* gpu_in_cut_yz,
              float* gpu_out_cut_xz,
              float* gpu_out_cut_yz,
              uint width,
              uint height,
              int time_transformation_size,
              const holovibes::CompositeHSV& hsv_struct,
              const cudaStream_t stream);
*/