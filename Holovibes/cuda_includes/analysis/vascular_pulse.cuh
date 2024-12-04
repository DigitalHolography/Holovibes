/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include <cuda_runtime.h>
#include "compute_env.hh"

using holovibes::VesselnessFilterStruct;

void compute_first_correlation(float* output,
                               float* M0_ff_video_centered,
                               float* vascular_pulse,
                               int nnz_mask_vesslness_clean,
                               size_t length_video,
                               VesselnessFilterStruct& filter_struct_,
                               size_t image_size,
                               cudaStream_t stream);
void multiply_three_vectors(
    float* output, float* input1, float* input2, float* input3, size_t size, cudaStream_t stream);

void divide_constant(float* vascular_pulse, int value, size_t size, cudaStream_t stream);