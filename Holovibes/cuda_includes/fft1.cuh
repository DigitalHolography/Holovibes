/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

/*! \brief Find the right threads and block to call quadratic lens with and call it */
void fft1_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream);

/*! \brief Apply a lens and call an fft1 on the image
 *
 * \param lens the lens that will be applied to the image
 * \param plan the first paramater of cufftExecC2C that will be called on the image
 */
void fft_1(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const cuComplex* lens,
           const cufftHandle plan2D,
           const size_t frame_resolution,
           const cudaStream_t stream);