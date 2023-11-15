/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "cuComplex.h"

void complex_to_doppler(float* output,
                        const cuComplex* input,
                        float* gpu_moment_zero_out,
                        float* gpu_moment_two_out,
                        const ushort pmin,
                        const ushort pmax,
                        const size_t frame_res,
                        const cudaStream_t stream);
