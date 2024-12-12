/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include <cuda_runtime.h>
#include "compute_env.hh"

using holovibes::VesselnessFilterEnv;

void compute_first_correlation(float* const output,
                               float* const M0_ff_video_centered,
                               float* const vascular_pulse,
                               const int nnz_mask_vesslness_clean,
                               const size_t length_video,
                               const VesselnessFilterEnv& filter_struct_,
                               const size_t size,
                               const cudaStream_t stream);

void multiply_three_vectors(float* const output,
                            const float* const input1,
                            const float* const input2,
                            const float* const input3,
                            const size_t size,
                            const cudaStream_t stream);