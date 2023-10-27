/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

void normalize_filter(float* gpu_filter, size_t width, size_t height, const cudaStream_t stream);

void interpolate_filter(
    float* gpu_filter, size_t width, size_t height, size_t fd_width, size_t fd_height, const cudaStream_t stream);

void apply_filter(float* gpu_filter,
                  size_t width,
                  size_t height,
                  cuComplex* gpu_input,
                  size_t fd_width,
                  size_t fd_height,
                  const cudaStream_t stream);