/*! \file
 *
 * \brief Declaration of all apply mask functions
 */
#pragma once

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in out] in_out The buffer of images to modify
 * \param[in] mask The mask to apply to 'in_out'
 * \param[in] size The number of pixels in one frame of 'in_out'
 * \param[in] batch_size The number of frames of 'in_out'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(
    cuComplex* in_out, const cuComplex* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in out] in_out The buffer of images to modify
 * \param[in] mask The mask to apply to 'in_out'
 * \param[in] size The number of pixels in one frame of 'in_out'
 * \param[in] batch_size The number of frames of 'in_out'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(
    cuComplex* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in out] in_out The buffer of images to modify
 * \param[in] mask The mask to apply to 'in_out'
 * \param[in] size The number of pixels in one frame of 'in_out'
 * \param[in] batch_size The number of frames of 'in_out'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(float* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in] input The buffer of images to modify
 * \param[in] mask The mask to apply to 'input' stored in 'output'
 * \param[out] output The output buffer of the mask application
 * \param[in] size The number of pixels in one frame of 'input'
 * \param[in] batch_size The number of frames of 'input'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const cuComplex* input,
                const cuComplex* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in] input The buffer of images to modify
 * \param[in] mask The mask to apply to 'input' stored in 'output'
 * \param[out] output The output buffer of the mask application
 * \param[in] size The number of pixels in one frame of 'input'
 * \param[in] batch_size The number of frames of 'input'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const cuComplex* input,
                const float* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

/*! \brief This function applies a mask to a number of frames
 *
 * \param[in] input The buffer of images to modify
 * \param[in] mask The mask to apply to 'input' stored in 'output'
 * \param[out] output The output buffer of the mask application
 * \param[in] size The number of pixels in one frame of 'input'
 * \param[in] batch_size The number of frames of 'input'
 * \param[in] stream The CUDA stream on which to launch the operation.
 */
void apply_mask(const float* input,
                const float* mask,
                float* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream);

/*! \brief Computes the mean of the pixels inside the image only if the pixel is in the given mask.
 *  Calls a CUDA Kernel.
 *
 *  \param[in] input The input image on which the mask is applied and the mean of pixels is computed.
 *  \param[in] mask The mean will be computed only inside this mask.
 *  \param[out] pixels_mean Pointer to store the mean of the pixels inside the circle.
 *  \param[in] size The size of the image, e.g : width x height.
 *  \param[in] stream The CUDA stream on which to launch the operation.
 */
void get_mean_in_mask(
    const float* input, const float* mask, float* pixels_mean, const size_t size, const cudaStream_t stream);

/*! \brief Rescaling by substracting the given mean to each pixels inside the masks. Set the others pixels to 0.
 *  Calls a CUDA Kernel.
 *
 *  \param[out] output The output image on which the mask is applied and the pixels are rescaled.
 *  \param[in] input The input image to get the pixels.
 *  \param[in] mask The pixels are rescaled only inside this mask.
 *  \param[in] mean The mean substracted to the pixels.
 *  \param[in] size The size of the image, e.g : width x height.
 *  \param[in] stream The CUDA stream on which to launch the operation.
 */
void rescale_in_mask(
    float* output, const float* input, const float* mask, const float mean, size_t size, const cudaStream_t stream);

/*! \brief Rescaling by substracting the given mean to each pixels inside the masks. Set the others pixels to 0.
 *  Overload of `rescale_in_mask`, calling it with `input_output` for both input and output.
 *  Calls a CUDA Kernel.
 *
 *  \param[in out] input_output The image on which the mask is applied and the pixels are rescaled. Operations done in
 *  place.
 *  \param[in] mask The pixels are rescaled only inside this mask.
 *  \param[in] mean The mean substracted to the pixels.
 *  \param[in] size The size of the image, e.g : width x height.
 *  \param[in] stream The CUDA stream on which to launch the operation.
 */
void rescale_in_mask(float* input_output, const float* mask, const float mean, size_t size, const cudaStream_t stream);