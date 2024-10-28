/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void load_kernel_in_GPU(cuComplex* output, const float* kernel, const size_t frame_res, cudaStream_t stream);

float* kernel_add_padding(float* kernel, const int width, const int height, const int new_width, const int new_height);