/*! \file chart_mean_vessels.cuh
 *
 * \brief TODO
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "common.cuh"
#include "chart_point.hh"

using uint = unsigned int;

holovibes::ChartMeanVesselsPoint
get_sum_with_mask(const float* input, const float* mask, size_t size, float* sum_res, cudaStream_t stream);
