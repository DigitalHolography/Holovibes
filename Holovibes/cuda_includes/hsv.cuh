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

void from_distinct_components_to_interweaved_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream);

void from_interweaved_components_to_distinct_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream);

void apply_percentile_and_threshold(
    float* gpu_arr, uint frame_res, float low_threshold, float high_threshold, const cudaStream_t stream);

void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct);
