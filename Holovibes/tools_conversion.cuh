/*! \file
 *
 * Conversion functions between different data types and formats. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>
# include "frame_desc.hh"

/* CONVERSION FUNCTIONS */

/*! \brief This function allows to transform 8-bit data to its complex representation.
 *
 * The transformation is performed by putting the squareroot of each element's
 * value into the real and imaginary parts of the complex output.
 *
 * \param output Where to store the resulting complex data. Converted data
 * is being rescaled to 16-bit value ( * 2^16 / 2^8 ).
 * \param input 8-bit input data to convert.
 * \param size The number of elements to convert.
 * \param sqrt_array Pointer to a float array containing the results
 * of the square root function, starting from 0. Precondition : length(array) >= size
 */
__global__ void img8_to_complex(
  cufftComplex* output,
  const unsigned char* input,
  const unsigned int size);

/*! \brief This function allows to transform 16 bit data to its complex representation.
 *
 * The transformation is performed by putting the squareroot of each element's
 * value into the real and imaginary parts of the complex output.
 *
 * \param output Where to store the resulting complex data.
 * \param input 16-bit input data to convert.
 * \param size The number of elements to convert.
 * \param sqrt_array Pointer to a float array containing the results
 * of the square root function, starting from 0. Precondition : length(array) >= size
 */
__global__ void img16_to_complex(
  cufftComplex* output,
  const unsigned short* input,
  const unsigned int size);

/*! \brief This function allows to transform float 32 bits data to its complex representation.
*
* The transformation is performed by putting the squareroot of each element's
* value into the real and imaginary parts of the complex output.
*
* \param output Where to store the resulting complex data.
* \param input float 32-bit input data to convert.
* \param size The number of elements to convert.
* \param sqrt_array Pointer to a float array containing the results
* of the square root function, starting from 0. Precondition : length(array) >= size
*/
__global__ void float_to_complex(
	cufftComplex* output,
	const float* input,
	const unsigned int size);

/*! \brief Compute the modulus of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief Compute the squared modulus of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_squared_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief Compute argument (angle) of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_argument(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

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
void rescale_float(
  const float* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream);

/*! \brief Convert from big endian to little endian.
 * \param input The input data in big endian.
 * \param output Where to store the data converted in little endian.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream = 0);

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
  const float* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream = 0);


/*! \brief Convert data from complex data to unsigned short (16-bit).
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
void complex_to_ushort(
	const cufftComplex* input,
	unsigned int* output,
	const unsigned int size,
	cudaStream_t stream = 0);

/*! \brief Memcpy of a complex sized frame into another buffer */
void complex_to_complex(
	const cufftComplex* input,
	unsigned short* output,
	const unsigned int size,
	cudaStream_t stream = 0);

/*! \brief Cast buffer into real_buffer*/
void	buffer_size_conversion(char *real_buffer
	, const char *buffer
	, const camera::FrameDescriptor real_frame_desc
	, const camera::FrameDescriptor frame_desc);

/*! \brief Cuda Kernel for buffer_size_conversion*/
__global__ void	kernel_buffer_size_conversion(char *real_buffer
	, const char *buffer
	, const size_t frame_desc_width
	, const size_t frame_desc_height
	, const size_t real_frame_desc_width
	, const size_t area);

/*! \brief Cumulate images into one.
*
* \param input Input data should be contiguous.
* \param output Where to store the output.
* \param start Number of starting elmt.
* \param max_elmt Total number of elmt.
* \param nb_elmt Number of elmt that should be added.
* \param nb_pixel Number of pixel per image.
* \param stream The CUDA stream on which to launch the operation.
*/
void accumulate_images(
	const float *input,
	float *output,
	const size_t start,
	const size_t max_elmt,
	const size_t nb_elmt,
	const size_t nb_pixel,
	cudaStream_t stream = 0);

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
__global__ void kernel_accumulate_images(
	const float *input,
	float *output,
	const size_t start,
	const size_t max_elmt,
	const size_t nb_elmt,
	const size_t nb_pixel);

__global__ void kernel_normalize_images(
	float *image,
	const float max,
	const float min,
	const unsigned int size);

void rescale_float_unwrap2d(float *input,
	float *output,
	float *cpu_buffer,
	unsigned int frame_res,
	cudaStream_t stream = 0);

__global__ void kernel_rescale_argument(
	float *input,
	const unsigned int size);

void rescale_argument(
	float *input,
	const unsigned int frame_res,
	cudaStream_t stream = 0);