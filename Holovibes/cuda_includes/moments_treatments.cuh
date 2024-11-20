/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"

void add_frame_to_sum(const float* const new_frame, const size_t size, float* const sum_image, cudaStream_t stream);

void subtract_frame_from_sum(const float* const new_frame,
                             const size_t size,
                             float* const sum_image,
                             cudaStream_t stream);

void compute_mean(float* output, float* input, const size_t time_window, const size_t frame_size, cudaStream_t stream);

void compute_mean_1_2(
    float* const output, const float* const input, const size_t frame_size, const size_t frame_nb, cudaStream_t stream);

void image_centering(
    float* output, const float* m0_video, const float* m0_mean, const uint frame_size, const cudaStream_t stream);