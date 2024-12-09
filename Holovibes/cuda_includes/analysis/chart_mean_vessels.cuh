/*! \file chart_mean_vessels.cuh
 *
 * \brief TODO
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "common.cuh"

using uint = unsigned int;

float get_sum_with_mask(const float* input, const float* mask, size_t size, float* sum_res, cudaStream_t stream);