/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_desc.hh"

/**
 * \brief Compute Binarisation with Otsu threshold
 *
 * \param[in out] input The image to process
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void computeBinariseOtsu(float* input, const size_t width, const size_t height, const cudaStream_t stream);

/*! \brief Compute Binarisation with Otsu threshold and bradley method
 *
 * \param[in] d_image Input data should be contiguous
 * \param[out] d_output Where to store the output
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] window_size //TODO
 * \param[in] local_threshold_factor //TODO
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void computeBinariseOtsuBradley(float* d_image,
                                float*& d_output,
                                const size_t width,
                                const size_t height,
                                const int window_size,
                                const float local_threshold_factor,
                                const cudaStream_t stream);

/*! \brief Normalistation of \d_image
 *
 * \param[in out] d_image Data should be contiguous
 * \param[in] min Minimal value of \d_image
 * \param[in] max Maximal value of \d_image
 * \param[in] size Size of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void normalise(float* d_image, float min, float max, const size_t size, const cudaStream_t stream);
