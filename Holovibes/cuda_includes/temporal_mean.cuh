/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"

// void initialize_all_image(float** all_images, const int frame_size, const int time_window, const cudaStream_t
// stream);
void temporal_mean(float* input_output,
                   int* current_image,
                   float* image_buffer,
                   float* image_sum,
                   const int time_window,
                   const uint frame_size,
                   const cudaStream_t stream);
