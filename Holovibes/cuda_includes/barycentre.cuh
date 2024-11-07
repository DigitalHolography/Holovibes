/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void compute_barycentre(float* output,
                        float* temporal_mean_img,
                        float* temporal_mean_video,
                        size_t size,
                        cudaStream_t stream);