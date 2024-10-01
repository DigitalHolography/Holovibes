/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

void apply_filter(float* gpu_filter2d_mask,
                  float* gpu_input_filter_mask,
                  const float* input_filter,
                  size_t width,
                  size_t height,
                  const cudaStream_t stream);
