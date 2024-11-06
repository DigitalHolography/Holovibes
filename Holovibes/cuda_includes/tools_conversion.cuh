/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include <nppdefs.h>

#include "frame_desc.hh"

/* CONVERSION FUNCTIONS */

/*! \brief Compute the modulus of complex image(s).
 *
 * \param output[out] Where to store the output.
 * \param input[in] Input data should be contiguous.
 * \param pmin Minimum index to compute on
 * \param pmax Maximum index to compute on
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_modulus(float* output,
                        const cuComplex* input,
                        const ushort pmin,
                        const ushort pmax,
                        const size_t size,
                        const cudaStream_t stream);

/*! \brief Compute the squared modulus of complex image(s).
 *
 * \param output[out] Where to store the output.
 * \param input[in] Input data should be contiguous.
 * \param pmin Minimum index to compute on
 * \param pmax Maximum index to compute on
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_squared_modulus(float* output,
                                const cuComplex* input,
                                const ushort pmin,
                                const ushort pmax,
                                const size_t size,
                                const cudaStream_t stream);

/*! \brief Compute argument (angle) of complex image(s).
 *
 * \param output[out] Where to store the output.
 * \param input[in] Input data should be contiguous.
 * \param pmin Minimum index to compute on
 * \param pmax Maximum index to compute on
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_argument(float* output,
                         const cuComplex* input,
                         const ushort pmin,
                         const ushort pmax,
                         const size_t size,
                         const cudaStream_t stream);

/*! \brief Copy the rescaled angle value of each element of the input.
 *
 * The function searches the minimum and maximum values among
 * the *size* elements, and rescales all elements so that the minimum
 * becomes "zero", and the maximum "2^16" on a 16-bit scale.
 *
 * \param output[out] Where to store the rescaled result.
 * \param input[in] Angles values.
 * \param pmin Minimum index to compute on
 * \param pmax Maximum index to compute on
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void rescale_float(float* output, const float* input, const size_t size, const cudaStream_t stream);

/*! \brief Convert from big endian to little endian.
 * \param output[out] Where to store the data converted in little endian.
 * \param input[in] The input data in big endian.
 * \param batch_size The batch size of the input.
 * \param frame_res The resolution (number of pixel) of a frame
 * \param stream The CUDA stream on which to launch the operation.
 */
void endianness_conversion(
    ushort* output, const ushort* input, const uint batch_size, const size_t frame_res, const cudaStream_t stream);

/*! \brief Convert data from float to unsigned short (16-bit).
 *
 * The input data shall be restricted first to the range [0; 2^16 - 1],
 * by forcing every negative  value to 0 and every positive one
 * greater than 2^16 - 1 to 2^16 - 1.
 * Then it is truncated to unsigned short data type.
 *
 * \param output[out] Where to store the data converted in unsigned short.
 * \param input[in] The input floating-point data.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void float_to_ushort(
    ushort* const output, const float* const input, const size_t size, const cudaStream_t stream, const uint shift = 0);

void float_to_ushort_normalized(ushort* const output, const float* const input, const size_t size, cudaStream_t stream);

void ushort_to_uchar(uchar* output, const ushort* input, const size_t size, const cudaStream_t stream);

/*! \brief Converts and tranfers data from input_queue to gpu_input_buffer
 *
 * Template call between .cc and .cu(h) almost never works and the data has to
 * be casted anyway (switch depth void* cast) So we chose to use template inside
 * the cu/cuh to factorize code but keep the void* between the cc and cu
 *
 * \param output The gpu input buffer.
 * \param input The input queue.
 * \param frame_res The total size of a frame (width * height).
 * \param batch_size The size of the batch to transfer.
 * \param depth The pixel depth.
 * \param stream The CUDA stream on which to launch the operation.
 */
void input_queue_to_input_buffer(void* const output,
                                 const void* const input,
                                 const size_t frame_res,
                                 const int batch_size,
                                 const camera::PixelDepth depth,
                                 const cudaStream_t stream);

/**
 * \brief Transfers data from a float buffer to another float buffer.
 *
 * Essentially the same function as above, but without the conversion.
 * Is used to dequeue moments from the input queue to a temporary buffer.
 *
 * \param output The gpu input buffer.
 * \param input The input queue.
 * \param frame_res The total size of a frame (width * height).
 * \param batch_size The size of the batch to transfer.
 * \param depth The pixel depth. Should be only Bits32 but is there just in case.
 * \param stream The CUDA stream on which to launch the operation.
 */
void input_queue_to_input_buffer_floats(void* const output,
                                        const void* const input,
                                        const size_t frame_res,
                                        const int batch_size,
                                        const camera::PixelDepth depth,
                                        const cudaStream_t stream);

/*! \brief Cumulate images into one.
 *
 * \param output[out] Where to store the output.
 * \param input[in] Input data should be contiguous.
 * \param end End of the queue. The most recent element in the queue
 * \param max_elmt Total number of elmt.
 * \param nb_elmt Number of elmt that should be added.
 * \param nb_pixel Number of pixel per image.
 * \param stream The CUDA stream on which to launch the operation.
 */
void accumulate_images(float* output,
                       const float* input,
                       const size_t end,
                       const size_t max_elmt,
                       const size_t nb_elmt,
                       const size_t nb_pixel,
                       const cudaStream_t stream);

/*! \brief Kernel to cumulate images into one.
 *
 * \param output[out] Where to store the output.
 * \param input[in] Input data should be contiguous.
 * \param start Number of starting elmt.
 * \param max_elmt Total number of elmt.
 * \param nb_elmt Number of elmt that should be added.
 * \param nb_pixel Number of pixel per image.
 * \param stream The CUDA stream on which to launch the operation.
 */
__global__ void kernel_accumulate_images(float* output,
                                         const float* input,
                                         const size_t start,
                                         const size_t max_elmt,
                                         const size_t nb_elmt,
                                         const size_t nb_pixel);

void normalize_complex(cuComplex* image, const size_t size, const cudaStream_t stream);

void rescale_float_unwrap2d(
    float* output, float* input, float* cpu_buffer, size_t frame_res, const cudaStream_t stream);

void convert_frame_for_display(void* output,
                               const void* input,
                               const size_t size,
                               const camera::PixelDepth depth,
                               const ushort shift,
                               const cudaStream_t stream);

void float_to_complex(cuComplex* output, const float* input, size_t size, const cudaStream_t stream);

/*!
 * \brief Convert a buffer filled with complex values into real values using the modulus. The function will
 * only convert from index f_start to index f_end.
 *
 * \param[out] output     Where to store the result. Same size as the input
 * \param[in]  input      The input buffer of size frame_res and of depth of at least `f_end`.
 * \param[in]  frame_res  The resolution of a single image
 * \param[in]  f_start    The start index
 * \param[in]  f_end      The end index
 * \param[in]  stream     The cuda stream
 */
void complex_to_modulus_moments(float* output,
                                const cuComplex* input,
                                const size_t frame_res,
                                const ushort f_start,
                                const ushort f_end,
                                const cudaStream_t stream);