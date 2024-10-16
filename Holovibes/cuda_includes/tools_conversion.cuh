/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include <nppdefs.h>

/* CONVERSION FUNCTIONS */

/*! \brief Compute the modulus of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
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
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
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
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
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
 * \param input Angles values.
 * \param output Where to store the rescaled result.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void rescale_float(const float* input, float* output, const size_t size, const cudaStream_t stream);

/*! \brief Convert from big endian to little endian.
 * \param input The input data in big endian.
 * \param output Where to store the data converted in little endian.
 * \param frame_res The resolution (number of pixel) of a frame
 * \param stream The CUDA stream on which to launch the operation.
 */
void endianness_conversion(
    const ushort* input, ushort* output, const uint batch_size, const size_t frame_res, const cudaStream_t stream);

/*! \brief Convert data from float to unsigned short (16-bit).
 *
 * The input data shall be restricted first to the range [0; 2^16 - 1],
 * by forcing every negative  value to 0 and every positive one
 * greater than 2^16 - 1 to 2^16 - 1.
 * Then it is truncated to unsigned short data type.
 *
 * \param input The input floating-point data.
 * \param output Where to store the data converted in unsigned short.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void float_to_ushort(
    const float* const input, ushort* const output, const size_t size, const cudaStream_t stream, const uint shift = 0);

void float_to_ushort_normalized(const float* const input, ushort* const output, const size_t size, cudaStream_t stream);

void ushort_to_uchar(const ushort* input, uchar* output, const size_t size, const cudaStream_t stream);

/*! \brief Converts and tranfers data from input_queue to gpu_spatial_transformation_queue
 *
 * Template call between .cc and .cu(h) almost never works and the data has to
 * be casted anyway (switch depth void* cast) So we chose to use template inside
 * the cu/cuh to factorize code but keep the void* between the cc and cu
 *
 * \param output The gpu_spatial_transformation_queue.
 * \param input The input queue.
 * \param frame_res The total size of a frame (width * height).
 * \param batch_size The size of the batch to transfer.
 * \param depth The pixel depth (uchar : 1, ushort : 2, float : 4).
 */
void input_queue_to_input_buffer(void* const output,
                                 const void* const input,
                                 const size_t frame_res,
                                 const int batch_size,
                                 const uint depth,
                                 const cudaStream_t stream);

/*! \brief Cumulate images into one.
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param end End of the queue. The most recent element in the queue
 * \param max_elmt Total number of elmt.
 * \param nb_elmt Number of elmt that should be added.
 * \param nb_pixel Number of pixel per image.
 * \param stream The CUDA stream on which to launch the operation.
 */
void accumulate_images(const float* input,
                       float* output,
                       const size_t end,
                       const size_t max_elmt,
                       const size_t nb_elmt,
                       const size_t nb_pixel,
                       const cudaStream_t stream);

/*! \brief Kernel to cumulate images into one.
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param start Number of starting elmt.
 * \param max_elmt Total number of elmt.
 * \param nb_elmt Number of elmt that should be added.
 * \param nb_pixel Number of pixel per image.
 * \param stream The CUDA stream on which to launch the operation.
 */
__global__ void kernel_accumulate_images(const float* input,
                                         float* output,
                                         const size_t start,
                                         const size_t max_elmt,
                                         const size_t nb_elmt,
                                         const size_t nb_pixel);

void normalize_complex(cuComplex* image, const size_t size, const cudaStream_t stream);

void rescale_float_unwrap2d(
    float* input, float* output, float* cpu_buffer, size_t frame_res, const cudaStream_t stream);

void convert_frame_for_display(const void* input,
                               void* output,
                               const size_t size,
                               const uint depth,
                               const ushort shift,
                               const cudaStream_t stream);

void float_to_complex(cuComplex* output, const float* input, size_t size, const cudaStream_t stream);
