/**
 * @file angular_spectrum.cu
 *
 * @brief Contains the implementation of the angular spectrum algorithm for image processing.
 * The angular spectrum algorithm is used to simulate the propagation of light through a lens.
 *
 * Usage:
 * - To calculate the angular spectrum for a lens, use @ref angular_spectrum_lens "angular_spectrum_lens".
 * - To implement the angular spectrum algorithm, use @ref angular_spectrum "angular_spectrum".
 * Code example:
 * ```cpp
 * // Calculate the angular spectrum for a lens
 * cuComplex* lens;
 * uint lens_side_size;
 * uint frame_height;
 * uint frame_width;
 * float lambda;
 * float z;
 * float pixel_size;
 * cudaStream_t stream;
 * angular_spectrum_lens(lens, lens_side_size, frame_height, frame_width, lambda, z, pixel_size, stream);
 *
 * // Implement the angular spectrum algorithm
 * cuComplex* input;
 * cuComplex* output;
 * uint batch_size;
 * cuComplex* mask_output;
 * bool store_frame;
 * cufftHandle plan2d;
 * FrameDescriptor fd;
 * angular_spectrum(input, output, batch_size, lens, mask_output, store_frame, plan2d, fd, stream);
 * ```
 */

#pragma once

#include "common.cuh"

/*!
 * \brief Calculate the angular spectrum for a lens.
 *
 * This function calculates the angular spectrum for a lens. If the frame is not
 * square, it allocates memory for a square lens, performs the calculation, and then copies the
 * result to the lens array.
 *
 * The output pointer can be another complex buffer or the same as input buffer.
 *
 * \param[in,out] lens The lens array.
 * \param[in] lens_side_size The size of the lens' both sides, as it is a square.
 * \param[in] frame_height The height of the frame.
 * \param[in] frame_width The width of the frame.
 * \param[in] lambda The wavelength.
 * \param[in] z The distance of the lens.
 * \param[in] pixel_size The size of the pixel used by the kernel.
 * \param[in] stream The input (and output) stream ; the data.
 */
void angular_spectrum_lens(cuComplex* __restrict__ lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream);

/*!
 * \brief Implement the angular spectrum algorithm.
 *
 * This function implements the angular spectrum algorithm. It uses the fast
 * Fourier transform (FFT) to transform the input array into the Fourier domain, applies a mask
 * using the apply_mask function, and then transforms the input array back into the spatial domain.
 * It then divides each element of the input array by the size of the frame.
 *
 * The output pointer can be another complex buffer or the same as input buffer.
 *
 * \param[in,out] input The input array.
 * \param[out] output The output array.
 * \param[in] batch_size The batch size.
 * \param[in] lens The lens array.
 * \param[out] mask_output The mask output array.
 * \param[in] store_frame A flag indicating whether to store the frame.
 * \param[in] plan2d The CUFFT plan, first paramater of cufftExecC2C that will be called on the image.
 * \param[in] fd The frame descriptor.
 * \param[in] stream The operation stream.
 */
void angular_spectrum(cuComplex* __restrict__ input,
           cuComplex* __restrict__ output,
           const uint batch_size,
           const cuComplex* lens,
           cuComplex* mask_output,
           bool store_frame,
           cufftHandle plan2d,
           const camera::FrameDescriptor& fd,
           const cudaStream_t stream);
