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
  const unsigned int size,
  const float* sqrt_array);

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
  const unsigned int size,
  const float* sqrt_array);

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
	const unsigned int size,
	const float* sqrt_array);

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

/*Todo:*/

void complex_to_ushort(
	const cufftComplex* input,
	unsigned int* output,
	const unsigned int size,
	cudaStream_t stream = 0);