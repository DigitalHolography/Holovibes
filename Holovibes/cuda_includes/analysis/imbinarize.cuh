#pragma once

float otsuthresh(float* counts, cudaStream_t stream);

float otsu_compute_threshold(float* input, float* histo_buffer_d, const size_t size, cudaStream_t stream);

void apply_binarisation(
    float* input_output, float threshold, const size_t width, const size_t height, const cudaStream_t stream);