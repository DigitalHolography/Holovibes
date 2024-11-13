/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void compute_first_correlation(float* output, float* image_centered, float* vascular_pulse, int nnz_mask_vesslness_clean, size_t length_video, size_t image_size, cudaStream_t stream);    // Size here is future time window
