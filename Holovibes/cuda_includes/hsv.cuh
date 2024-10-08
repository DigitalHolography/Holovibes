/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"
typedef unsigned int uint;

enum HSV
{
    H = 0,
    S = 1,
    V = 2
};

void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct,
         bool checked);

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
