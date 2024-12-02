/*! \file otsu.cuh
 *
 * \brief function for otsu compute, otsu is a method for binarisation, you can check :
 * https://en.wikipedia.org/wiki/Otsu%27s_method
 */
#pragma once

#include "frame_desc.hh"

using uint = unsigned int;

/**
 * \brief Compute Binarisation with Otsu threshold
 *
 * \param[in out] input_output The image to process
 * \param[in] threshold
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void compute_binarise_otsu(
    float* input_output, float threshold, const size_t width, const size_t height, const cudaStream_t stream);

/*! \brief Compute Binarisation with Otsu threshold and bradley method
 *
 * \param[out] output_d Where to store the output
 * \param[out] histo_buffer_d gpu buffer for histogram
 * \param[in] input_d Input data should be contiguous
 * \param[in] threshold_d GPU float use inside
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] window_size size of the windows
 * \param[in] local_threshold_factor local threshold factor
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void compute_binarise_otsu_bradley(float* output_d,
                                   uint* histo_buffer_d,
                                   const float* input_d,
                                   float* threshold_d,
                                   const size_t width,
                                   const size_t height,
                                   const int window_size,
                                   const float local_threshold_factor,
                                   const cudaStream_t stream);

/*!
 * \brief get otsu threshold use for binarization of grey image
 *
 * \param[in] image_d Input data should be contiguous
 * \param[out] histo_buffer_d gpu buffer for histogram
 * \param[in] threshold_d GPU float use inside
 * \param[in] size Size of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 * \return ostus threshold
 */
float otsu_threshold(
    const float* image_d, uint* histo_buffer_d, float* threshold_d, int size, const cudaStream_t stream);
