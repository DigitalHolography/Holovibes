/*! \file moments_treatments.cuh
 *
 * \brief Contains functions dedicated to the mean and centering computations on moments
 */
#pragma once

#include <cuda_runtime.h>

/*!
 * \brief Add the input frame to the input_output frame buffer sum image. For each pixel the input pixel value is
 * accumulated to the pixel of input_output.
 *
 * This function was originally made for the CircleVideoBuffer class, used when we want to "enqueue" a new frame to the
 * total sum
 *
 * \param[in out] input_output The sum image input output
 * \param[in] input The input frame to add inside input_output
 * \param[in] size The size of a single frame
 * \param[in] stream The CUDA stream to use
 */
void add_frame_to_sum(float* const input_output, const float* const input, const size_t size, cudaStream_t stream);

/*!
 * \brief Subtracts the input frame from the input_output frame buffer sum image. For each pixel the input pixel value
 * is subtracted from the pixel of input_output.
 *
 * This function was originally made for the CircleVideoBuffer class, used when the buffer is full and need to "dequeue"
 * the oldest frame to have room for a new one, avoiding us the need to recompute the whole mean
 *
 * \param[in out] input_output The sum image input output
 * \param[in] input The The input frame to add inside input_output
 * \param[in] size The size of a single frame
 * \param[in] stream The CUDA stream to use
 */
void subtract_frame_from_sum(float* const input_output,
                             const float* const input,
                             const size_t size,
                             cudaStream_t stream);

/*!
 * \brief Computes the mean from the already calculated sum of each elements in the input buffer.
 *
 * This function was originally made for the CircleVideoBuffer class, allowing to compute the mean once from a sum image
 * and store it for later
 *
 * \param[out] output The output buffer
 * \param[in] input The input sum image buffer
 * \param[in] time_window The divider to divide each pixel with
 * \param[in] frame_size The size of a frame
 * \param[in] stream The CUDA stream to use
 */
void compute_mean(float* const output,
                  const float* const input,
                  const size_t time_window,
                  const size_t frame_size,
                  cudaStream_t stream);

/*!
 * \brief Centers the frames of a video by subtracting the mean frame using a CUDA kernel.
 *
 * This function centers the frames of a video by subtracting the mean frame from each frame. It configures and launches
 * a CUDA kernel to perform the centering operation. The function uses the provided CUDA stream for asynchronous
 * execution.
 *
 * \param [out] output Pointer to the output array where the centered frames will be stored.
 * \param [in] m0_video Pointer to the input video array containing the frames to be centered.
 * \param [in] m0_mean Pointer to the mean frame array.
 * \param [in] frame_size The number of elements in each frame.
 * \param [in] length_video The number of frames in the video.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the frame size and video length.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch.
 */
void image_centering(float* output,
                     const float* m0_video,
                     const float* m0_mean,
                     const size_t frame_size,
                     const size_t length_video,
                     const cudaStream_t stream);