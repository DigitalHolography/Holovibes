/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

# include "Common.cuh"

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
__global__
void img8_to_complex(cuComplex	*output,
					const uchar	*input,
					const uint	size);

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
__global__
void img16_to_complex(cuComplex		*output,
					const ushort	*input,
					const uint		size);

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
__global__
void float_to_complex(cuComplex	*output,
					const float	*input,
					const uint	size);

/*! \brief Compute the modulus of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_modulus(const cuComplex	*input,
						float			*output,
						const uint		size,
						cudaStream_t	stream = 0);

/*! \brief Compute the squared modulus of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_squared_modulus(const cuComplex	*input,
								float			*output,
								const uint		size,
								cudaStream_t	stream = 0);

/*! \brief Compute argument (angle) of complex image(s).
 *
 * \param input Input data should be contiguous.
 * \param output Where to store the output.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void complex_to_argument(const cuComplex	*input,
						float				*output,
						const uint			size,
						cudaStream_t		stream = 0);

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
void rescale_float(const float	*input,
				float			*output,
				const uint		size,
				cudaStream_t	stream);

/*! \brief Convert from big endian to little endian.
 * \param input The input data in big endian.
 * \param output Where to store the data converted in little endian.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 */
void endianness_conversion(const ushort	*input,
						ushort			*output,
						const uint		size,
						cudaStream_t	stream = 0);

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
void float_to_ushort(const float	*input,
					void			*output,
					const uint		size,
					const float		depth,
					cudaStream_t	stream = 0);


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
void complex_to_ushort(const cuComplex	*input,
					uint				*output,
					const uint			size,
					cudaStream_t		stream = 0);

/*! \brief Memcpy of a complex sized frame into another buffer */
void complex_to_complex(const cuComplex	*input,
						ushort			*output,
						const uint		size,
						cudaStream_t	stream = 0);

/*! \brief Cast buffer into real_buffer*/
void	buffer_size_conversion(char					*real_buffer,
							const char				*buffer,
							const camera::FrameDescriptor	real_fd,
							const camera::FrameDescriptor	fd);

/*! \brief Cuda Kernel for buffer_size_conversion*/
__global__
void	kernel_buffer_size_conversion(char			*real_buffer,
									const char		*buffer,
									const size_t	fd_width,
									const size_t	fd_height,
									const size_t	real_fd_width,
									const size_t	area);

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
void accumulate_images(const float	*input,
					float			*output,
					const size_t	start,
					const size_t	max_elmt,
					const size_t	nb_elmt,
					const size_t	nb_pixel,
					cudaStream_t	stream = 0);

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
__global__
void kernel_accumulate_images(const float	*input,
							float			*output,
							const size_t	start,
							const size_t	max_elmt,
							const size_t	nb_elmt,
							const size_t	nb_pixel);

__global__
void kernel_normalize_images(float		*image,
							const float	max,
							const float	min,
							const uint	size);
__global__
void kernel_normalize_complex(cuComplex		*image,
							const uint	size);
void normalize_complex(cuComplex		*image,
						const uint	size);

void rescale_float_unwrap2d(float			*input,
							float			*output,
							float			*cpu_buffer,
							uint			frame_res,
							cudaStream_t	stream = 0);

__global__
void kernel_rescale_argument(float		*input,
							const uint	size);

void rescale_argument(float			*input,
					const uint		frame_res,
					cudaStream_t	stream = 0);
