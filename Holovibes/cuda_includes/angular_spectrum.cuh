/*! \file angular_spectrum.cuh
 *
 * \brief Declaration of angular spectrum compute functions.
 */
#pragma once

#include "common.cuh"
#include "frame_desc.hh"

/*! \brief Compute the lens to apply to the image during the angular spectrum process.
 *  The `x_step` and `y_step` params ar equals for now since they are computed from pixel_size.
 *  However, for the futur we may want them to be unequal.
 *
 * \param[out] output The buffer to store the lens.
 * \param[in] Nx The width of the buffer.
 * \param[in] Ny The height of the buffer.
 * \param[in] z The Z distance setting
 * \param[in] lambda The lambda setting
 * \param[in] x_step The pixel width.
 * \param[in] y_step The pixel height.
 * \param[in] stream The input (and output) stream
 */
void angular_spectrum_lens(cuFloatComplex* output,
                           const int Nx,
                           const int Ny,
                           const float z,
                           const float lambda,
                           const float x_step,
                           const float y_step,
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
