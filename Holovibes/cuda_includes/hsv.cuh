/*! \file
 *
 * \brief declaration of hsv function
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

/*!
 * \brief Create rgb color by using hsv computation and then converting to rgb
 *
 * \param d_input complex input buffer, on gpu side, size = width * height * time_transformation_size
 * \param d_output float output buffer, on gpu side, size = width * height * 3
 * \param width Width of the frame
 * \param height Height of the frame
 * \param stream Cuda stream used
 * \param time_transformation_size Depth of the frame cube
 * \param hsv_struct Struct containing all the UI parameters
 * \param checked Boolean to know if the user wants to use the checked version of the function
 */
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
