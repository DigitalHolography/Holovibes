#pragma once
#include "common.cuh"

void apply_flat_field_correction(
    float* input_output, const uint width, const float gw, const float borderAmount, const cudaStream_t stream);
