/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void compute_first_correlation(float* output, float* vascular_pulse, int nnz_mask_vesslness_clean, size_t size, cudaStream_t stream);
