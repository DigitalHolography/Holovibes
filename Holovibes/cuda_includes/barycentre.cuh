/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void compute_multiplication(float* output, float* A, float* B, size_t size, cudaStream_t stream);

int compute_barycentre_circle_mask(float* output,
                                    float *input,
                                    size_t size,
                                    cudaStream_t stream, 
                                    int CRV_index = -1);