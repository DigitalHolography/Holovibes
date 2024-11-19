/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuComplex.h"
#include "composite_struct.hh"

void segment_vessels(
    float* output, float* R_VascularPulse, float* mask_vesselness_clean, uint size, cudaStream_t stream);

void compute_first_mask_artery(float* output, float* input, uint size, cudaStream_t stream);

void compute_first_mask_vein(float* output, float* input, uint size, cudaStream_t stream);

void negation(float* input_output, uint size, cudaStream_t stream);