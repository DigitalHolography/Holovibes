/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include "common.cuh"

void compute_multiplication(float* output, float* A, float* B, size_t size, uint depth, cudaStream_t stream);

void compute_multiplication_mean(float* output, float* A, float* B, size_t size, uint depth, cudaStream_t stream);

int compute_barycentre_circle_mask(float* output, float* input, size_t size, cudaStream_t stream, int CRV_index = -1);