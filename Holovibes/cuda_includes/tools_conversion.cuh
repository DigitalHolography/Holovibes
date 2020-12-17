/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
                        const uint size,
                        const cudaStream_t stream = 0);

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
                                const uint size,
                                const cudaStream_t stream = 0);

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
                         const uint size,
                         const cudaStream_t stream = 0);

/*! Copy the rescaled angle value of each element of the input.
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
void rescale_float(const float* input,
                   float* output,
                   const uint size,
                   const cudaStream_t stream = 0);

/*! \brief Convert from big endian to little endian.
 * \param input The input data in big endian.
 * \param output Where to store the data converted in little endian.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void endianness_conversion(const ushort* input,
                           ushort* output,
                           const uint batch_size,
                           const uint size,
                           const cudaStream_t stream = 0);

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
void float_to_ushort(const float* const input,
                     ushort* const output,
                     const uint size,
                     const uint shift = 0,
                     const cudaStream_t stream = 0);

void ushort_to_uchar(const ushort* input,
                     uchar* output,
                     const uint size,
                     const cudaStream_t stream = 0);

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
 * \param current_queue_index The current index of the input queue.
 * \param queue_size The total size of the input queue (max number of elements).
 * \param depth The pixel depth (uchar : 1, ushort : 2, float : 4).
 */
void input_queue_to_input_buffer(void* output,
                                 void* input,
                                 const uint frame_res,
                                 const int batch_size,
                                 const uint current_queue_index,
                                 const uint queue_size,
                                 const uint depth);

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
                       const cudaStream_t stream = 0);

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

void normalize_complex(cuComplex* image,
                       const uint size,
                       const cudaStream_t stream = 0);

void rescale_float_unwrap2d(float* input,
                            float* output,
                            float* cpu_buffer,
                            uint frame_res,
                            const cudaStream_t stream = 0);

void convert_frame_for_display(const void* input,
                               void* output,
                               const uint size,
                               const uint depth,
                               const ushort shift);

/*! \brief Converts frame in complex
** \param frame_res Size of frame in input
*/
void frame_to_complex(void* input,
                      cufftComplex* output,
                      const uint frame_res,
                      const uint depth);