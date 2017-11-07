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

#include "Rectangle.hh"

/// Extract the part of *input described by frame
void extract_frame(const float	*input,
				float			*output,
				const uint		input_w,
				const holovibes::units::RectFd&	frame);

/// Resize the image
void gpu_resize(const float		*input,
				float			*output,
				QPoint			old_size,
				QPoint			new_size,
				cudaStream_t	stream = 0);

/// Mirrors the image inplace on both axis
void rotation_180(float				*frame,
					QPoint			size,
					cudaStream_t	stream = 0);

/// A(x, y) = sum[i<=x, j<=y] ( B(i, j) )
void sum_left_top(const float		*input,
					float			*output,
					QPoint			size,
					cudaStream_t	stream = 0);

/// A(x, y) = sum[i<=x, j<=y] ( A(i, j) )
void sum_left_top_inplace(float			*input,
							QPoint			size,
							cudaStream_t	stream = 0);

/// A(x, y) = sum[i<=x, j<=y] ( A(i, j)^2 )
void sum_inplace_squared(float			*input,
						QPoint			size,
						cudaStream_t	stream = 0);

/// output = sum(convolution) - ((sum(a) * sum(b)) / overlap)
/// 
/// Output written inplace in sum_convolution
void compute_numerator(const float		*sum_a,
					const float			*sum_b,
					float				*sum_convolution,
					QPoint				size,
					cudaStream_t		stream = 0);

/// output = sum(A^2) - (sum(A)^2 / overlap)
/// 
/// Output written inplace in matrix
void sum_squared_minus_square_sum(
					float			*matrix,
					const float		*sum_squared,
					QPoint			size,
					cudaStream_t	stream = 0);

/// output = numerator / sqrt( denominator1 * denominator2)
/// 
/// Output written inplace in numerator
void correlation(float			*numerator,
				const float		*denominator1,
				const float		*denominator2,
				QPoint			size,
				cudaStream_t	 = 0);

/// Pads each line with 0, inplace
///
/// [a b c d; 0 0 0 0] => [a b 0 0; c d 0 0]
void pad_frame(float		*frame,
				QPoint		old_size,
				QPoint		new_size,
				cudaStream_t	stream = 0);