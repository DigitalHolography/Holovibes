/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

void apply_filter(float* gpu_filter, cuComplex* gpu_input, size_t frame_res, const cudaStream_t stream);