/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

/*! \brief takes input complex buffer and computes a p frame that is stored at output pointer.
 *
 * The output pointer can be another complex buffer or the same as input buffer.
 */
void fft2_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream);

/*! \brief takes input complex buffer and computes a p frame that is stored at output pointer.
 *
 * The output pointer can be another complex buffer or the same as input buffer.
 */
void fft_2(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const cuComplex* lens,
           cuComplex* mask_output,
           bool store_frame,
           const cufftHandle plan2d,
           const camera::FrameDescriptor& fd,
           const cudaStream_t stream);
