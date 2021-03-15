/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */
#pragma once
#include "compute_descriptor.hh"
#include "cuComplex.h"
typedef unsigned int uint;

void from_distinct_components_to_interweaved_components(
    const float* src, float* dst, size_t frame_res, const cudaStream_t stream);

void from_interweaved_components_to_distinct_components(
    const float* src, float* dst, size_t frame_res, const cudaStream_t stream);

void apply_percentile_and_threshold(float* gpu_arr,
                                    uint frame_res,
                                    float low_threshold,
                                    float high_threshold,
                                    const cudaStream_t stream);

void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const holovibes::ComputeDescriptor& cd,
         const cudaStream_t stream);