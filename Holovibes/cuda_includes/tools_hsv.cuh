/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"

typedef unsigned int uint;

void threshold_top_bottom(
    float* output, const float tmin, const float tmax, const uint frame_res, const cudaStream_t stream);

void apply_percentile_and_threshold(float* gpu_arr,
                                    uint frame_res,
                                    uint width,
                                    uint height,
                                    float low_threshold,
                                    float high_threshold,
                                    const cudaStream_t stream);

void rotate_hsv_to_contiguous_z(const cuComplex* gpu_input,
                                float* rotated_hsv_arr,
                                const uint frame_res,
                                const uint width,
                                const uint range,
                                const cudaStream_t stream);

void from_distinct_components_to_interweaved_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream);

void from_interweaved_components_to_distinct_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream);
