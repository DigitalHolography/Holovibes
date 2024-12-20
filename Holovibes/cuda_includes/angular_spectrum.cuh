/*! \file
 *
 * \brief Declaration of angular spectrum compute functions
 */
#pragma once

#include "common.cuh"

/*! \brief takes input complex buffer and computes a p frame that is stored at output pointer.
 *
 * The output pointer can be another complex buffer or the same as input buffer.
 *
 * \param lens The lens applied to the image
 * \param lens_side_size The size of the lens' both sides, as it is a square
 * \param frame_height Height of each frame
 * \param frame_width Width of each frame
 * \param lambda The lambda setting
 * \param z The Z distance setting
 * \param pixel_size Used by the kernel
 * \param stream The input (and output) stream ; the data
 */
void angular_spectrum_lens(cuComplex* lens,
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
 *
 * \param input Input data
 * \param output Output data
 * \param batch_size The number of images in a single batch
 * \param lens the lens that will be applied to the image
 * \param plan2D the first paramater of cufftExecC2C that will be called on the image
 * \param frame_resolution The total number of pixels in the image (width * height)
 * \param stream The operation stream
 */
void angular_spectrum(cuComplex* input,
                      cuComplex* output,
                      const uint batch_size,
                      const cuComplex* lens,
                      cuComplex* mask_output,
                      bool store_frame,
                      const cufftHandle plan2d,
                      const camera::FrameDescriptor& fd,
                      const cudaStream_t stream);
