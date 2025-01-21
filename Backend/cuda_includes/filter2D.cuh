/*! \file
 *
 * \brief Declaration of filter 2D functions
 */
#pragma once

#include "common.cuh"

/*!
 * \brief Apply a 2D filter to a batch of images
 *
 * \param[in] input The input image
 * \param[in] mask The mask to apply
 * \param[out] output The output image
 * \param[in] store_frame Whether to store the frame in the output
 * \param[in] batch_size The number of frames in the batch
 * \param[in] plan2d The 2D FFT plan
 * \param[in] width The width of the input image
 * \param[in] length The length of the input image
 * \param[in] stream The CUDA stream to use
 */
void filter2D(cuComplex* input,
              const float* mask,
              cuComplex* output,
              bool store_frame,
              const uint batch_size,
              const cufftHandle plan2d,
              const uint width,
              const uint length,
              const cudaStream_t stream);

/*!
 * \brief Update the mask of a 2D filter
 *
 * \param[in out] in_out The mask to update
 * \param[in] width The width of the mask
 * \param[in] height The height of the mask
 * \param[in] radius_low The low radius of the mask
 * \param[in] radius_high The high radius of the mask
 * \param[in] smooth_low The low smooth of the mask
 * \param[in] smooth_high The high smooth of the mask
 * \param[in] stream The CUDA stream to use
 */
void update_filter2d_circles_mask(float* in_out,
                                  const uint width,
                                  const uint height,
                                  const uint radius_low,
                                  const uint radius_high,
                                  const uint smooth_low,
                                  const uint smooth_high,
                                  const cudaStream_t stream);
