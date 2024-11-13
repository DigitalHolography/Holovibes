/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"

void add_frame_to_sum(int frame_size, float* oldest_frame, float* sum_images_, cudaStream_t stream);

void subtract_frame_from_sum(int frame_size, float* new_frame, float* sum_images_, cudaStream_t stream);

void temporal_mean(float* output,
                   float* input,
                   int* current_image,
                   float* image_buffer,
                   float* image_sum,
                   const int time_window,
                   const uint frame_size,
                   const cudaStream_t stream);

void image_centering(
    float* output, const float* m0_video, const float* m0_img, const uint frame_size, const cudaStream_t stream);
