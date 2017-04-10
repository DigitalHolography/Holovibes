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

/*! \brief Convert complex values to floating-point angles in [-pi; pi].
 *
 * Take complex data in cartesian form, and use conversion to polar
 * form to take the angle value of each element and store it
 * in a floating-point matrix. The resulting angles' values are bound
 * in [-pi; pi]. */
__global__
void kernel_extract_angle(const cuComplex	*input,
						float				*output,
						const size_t		size);

/*! Perform element-wise phase adjustment on a pixel matrix.
 *
 * \param pred Predecessor phase image.
 * \param cur Latest phase image.
 * \param output Where to store the unwrapped version of cur.
 * \param size Size of an image in pixels. */
__global__
void kernel_unwrap(const float	*pred,
				const float		*cur,
				float			*output,
				const size_t	size);

/*! Use the multiply-with-conjugate method to fill a float (angles) matrix.
 *
 * Computes cur .* conjugate(pred),
 * where .* is the element-wise complex-valued multiplication operation.
 * The angles of the resulting complex matrix are stored in output.
 * \param pred Predecessor complex image.
 * \param cur Latest complex image.
 * \param output The matrix which shall store ther resulting angles.
 * \param size The size of an image in pixels. */
__global__
void kernel_compute_angle_mult(const cuComplex	*pred,
							const cuComplex		*cur,
							float				*output,
							const size_t		size);

/*! Use the subtraction method to fill a float (angles) matrix.
*
* Computes cur - conjugate(pred). The angles of the resulting complex matrix
* are stored in output.
* \param pred Predecessor complex image.
* \param cur Latest complex image.
* \param output The matrix which shall store ther resulting angles.
* \param size The size of an image in pixels. */
__global__
void kernel_compute_angle_diff(const cuComplex	*pred,
							const cuComplex		*cur,
							float				*output,
							const size_t		size);

/*! Iterate over saved phase corrections and apply them to an image.
*
* \param data The image to be corrected.
* \param corrections Pointer to the beginning of the phase corrections buffer.
* \param image_size The number of pixels in a single image.
* \param history_size The number of past phase corrections used. */
__global__
void kernel_correct_angles(float		*data,
						const float		*corrections,
						const size_t	image_size,
						const size_t	history_size);

/*! Initialise fx, fy, z matrix for unwrap 2d.
*
* \param Matrix width.
* \param Matrix height.
* \param Matrix resolution.
* \param 1TF or 2TF result.
* \param fx buffer.
* \param fy buffer.
* \param z buffer. */

__global__
void kernel_init_unwrap_2d(const uint	width,
						const uint		height,
						const uint		frame_res,
						const float		*input,
						float			*fx,
						float			*fy,
						cuComplex		*z);

/*! \brief  Multiply each pixels of a complex frame value by a float.
**	Done for 2 complexes.
*/
__global__
void kernel_multiply_complexes_by_floats_(const float	*input1,
										const float		*input2,
										cuComplex		*output1,
										cuComplex		*output2,
										const uint		size);

/*! \brief  Multiply each pixels of two complexes frames value by a single complex.
*/
__global__
void kernel_multiply_complexes_by_single_complex(cuComplex		*output1,
												cuComplex		*output2,
												const cuComplex	input,
												const uint		size);

/*! \brief  Multiply each pixels of complex frames value by a single complex.
*/
__global__
void kernel_multiply_complex_by_single_complex(cuComplex	*output,
											const cuComplex	input,
											const uint		size);

/*! \brief  Get conjugate complex frame.
*/
__global__
void kernel_conjugate_complex(cuComplex	*output,
							const uint	size);

/*! \brief  Multiply a complex frames by a complex frame.
*/
__global__
void kernel_multiply_complex_frames_by_complex_frame(cuComplex		*output1,
													cuComplex		*output2,
													const cuComplex	*input,
													const uint		size);

/*! \brief  Multiply a complex frames by ratio from fx or fy and norm of fx and fy.
*/
__global__ void kernel_norm_ratio(const float	*input1,
								const float		*input2,
								cuComplex		*output1,
								cuComplex		*output2,
								const uint		size);

/*! \brief  Add two complex frames into one.
*/
__global__
void kernel_add_complex_frames(cuComplex	*output,
							const cuComplex	*input,
							const uint		size);

/*! \brief  Calculate phi for a frame.
*/
__global__
void kernel_unwrap2d_last_step(float		*output,
							const cuComplex	*input,
							const uint		size);
