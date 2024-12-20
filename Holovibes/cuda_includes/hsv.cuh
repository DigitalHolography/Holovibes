/*! \file
 *
 * \brief Declaration of hsv function
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
 * \param[in] d_input complex input buffer, on gpu side, size = width * height * time_transformation_size
 * \param[out] d_output float output buffer, on gpu side, size = width * height * 3
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream Cuda stream used
 * \param[in] time_transformation_size Depth of the frame cube
 * \param[in] hsv_struct Struct containing all the UI parameters
 * \param[in] checked Boolean to know if the user wants to use the checked version of the function
 */
void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct,
         bool checked);
