/*! \file
 *
 * \brief Declaration of apply filter function
 */
#pragma once

#include "common.cuh"

/*!
 * \brief Apply a filter to a 2D image
 *
 * \param[in] gpu_filter2d_mask The mask to apply
 * \param[in] gpu_input_filter_mask The input image
 * \param[in] input_filter The filter to apply
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] stream The CUDA stream to use
 */
void apply_filter(float* gpu_filter2d_mask,
                  float* gpu_input_filter_mask,
                  const float* input_filter,
                  size_t width,
                  size_t height,
                  const cudaStream_t stream);
