/*! \file segment_vessels.cuh
 *
 * \brief Contains all functions tied to the vessels segmentation for analysis
 */
#pragma once

#include <cuda_runtime.h>

typedef unsigned int uint;

void segment_vessels(float* const output,
                     float* const new_thresholds,
                     float* const R_VascularPulse,
                     const float* const mask_vesselness_clean,
                     const size_t size,
                     const float* thresholds,
                     const cudaStream_t stream);

void compute_first_mask_artery(float* const output,
                               const float* const input,
                               const size_t size,
                               const cudaStream_t stream);

void compute_first_mask_vein(float* const output,
                             const float* const input,
                             const size_t size,
                             const cudaStream_t stream);

void negation(float* const input_output, const size_t size, const cudaStream_t stream);