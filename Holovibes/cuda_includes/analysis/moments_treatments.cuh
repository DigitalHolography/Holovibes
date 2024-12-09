/*! \file moments_treatments.cuh
 *
 * \brief Contains functions dedicated to the mean and centering computations on moments
 */
#pragma once

#include <cuda_runtime.h>

// add_frame_to_sum and subtract_frame_from_sum are generic add and subtract two matrixes functions, we don't use
// cublasSgamm because we want to do it in place
// TODO: make these functions generic and put them in the right place, use cublasSgamm ? (by adding a computation buffer
// in circular_video_buffer)
void add_frame_to_sum(float* const input_output, const float* const input, const size_t size, cudaStream_t stream);

void subtract_frame_from_sum(float* const input_output,
                             const float* const input,
                             const size_t size,
                             cudaStream_t stream);

void compute_mean(float* const output,
                  const float* const input,
                  const size_t time_window,
                  const size_t frame_size,
                  cudaStream_t stream);

void compute_mean_1_2(
    float* const output, const float* const input, const size_t frame_size, const size_t frame_nb, cudaStream_t stream);

void image_centering(float* output,
                     const float* m0_video,
                     const float* m0_mean,
                     const size_t frame_size,
                     const size_t length_video,
                     const cudaStream_t stream);