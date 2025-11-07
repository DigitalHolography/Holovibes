/*! \file
 *
 * \brief Helpers dedicated to the delete twin image space transformation.
 */
#pragma once

#include "common.cuh"

/*!
 * \brief Apply a separable Gaussian blur on a batch of frames.
 *
 * \param input Pointer to the unwrapped phase data.
 * \param temp Intermediate buffer used for the horizontal pass.
 * \param output Output buffer receiving the blurred result.
 * \param kernel Gaussian coefficients.
 * \param radius Kernel radius (kernel size = 2 * radius + 1).
 * \param width Frame width.
 * \param height Frame height.
 * \param batch_size Number of frames in the batch.
 * \param stream CUDA stream.
 */
void gaussian_blur_batch(const float* input,
                         float* temp,
                         float* output,
                         const float* kernel,
                         int radius,
                         int width,
                         int height,
                         uint batch_size,
                         const cudaStream_t stream);

/*!
 * \brief Compute element-wise subtraction of two float buffers.
 *
 * \param output Destination buffer.
 * \param minuend Buffer containing the values to subtract from.
 * \param subtrahend Buffer containing the values to subtract.
 * \param element_count Number of elements to process.
 * \param stream CUDA stream.
 */
void subtract_arrays(
    float* output, const float* minuend, const float* subtrahend, size_t element_count, const cudaStream_t stream);

/*!
 * \brief Build a complex field from amplitude and phase information.
 *
 * \param output Destination complex buffer.
 * \param amplitude Amplitude buffer.
 * \param phase Phase buffer.
 * \param element_count Number of elements to process.
 * \param negate_phase When true, use -phase in the exponential.
 * \param stream CUDA stream.
 */
void combine_amplitude_phase(cuComplex* output,
                             const float* amplitude,
                             const float* phase,
                             size_t element_count,
                             bool negate_phase,
                             const cudaStream_t stream);
