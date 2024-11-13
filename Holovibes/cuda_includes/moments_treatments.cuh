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

void temporal_mean(float* output,
                   float* input,
                   int* current_image,
                   float* image_buffer,
                   float* image_sum,
                   const int time_window,
                   const uint frame_size,
                   const cudaStream_t stream);

void image_centering(
    float* output, const float* m0_img, const float* m0_video, const uint frame_size, const cudaStream_t stream);
