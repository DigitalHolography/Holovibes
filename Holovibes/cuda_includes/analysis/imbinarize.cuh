#pragma once

float otsuthresh(float* counts, cudaStream_t stream);

float otsu_compute(float* input, float* histo_buffer_d, const size_t size, cudaStream_t stream);