/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

void filter2D(cuComplex* input,
              const float* mask,
              cuComplex* output,
              bool store_frame,
              const uint batch_size,
              const cufftHandle plan2d,
              const uint width,
              const uint length,
              const cudaStream_t stream);

void update_filter2d_circles_mask(float* in_out,
                                  const uint width,
                                  const uint height,
                                  const uint radius_low,
                                  const uint radius_high,
                                  const uint smooth_low,
                                  const uint smooth_high,
                                  const cudaStream_t stream);