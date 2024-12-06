/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include <cuda_runtime.h>

void compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream);

int compute_barycentre_circle_mask(float* output,
                                   float* crv_circle_mask,
                                   float* input,
                                   size_t width,
                                   size_t height,
                                   cudaStream_t stream,
                                   int CRV_index = -1);