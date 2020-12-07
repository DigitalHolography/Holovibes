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

#include "cuComplex.h"
#include "cuda_runtime.h"

using uint = unsigned int;
using ushort = unsigned short;

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to modify in-place.
 * \param batch_size Number of images in input
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(float*		input,
				   const uint 	batch_size,
				   const uint		size_x,
				   const uint		size_y,
				   cudaStream_t	stream = 0);

void shift_corners(cuComplex *input,
				   const uint 	batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream = 0);

void shift_corners(float3 *input,
				   const uint 	batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream = 0);

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to shift.
 * \param output The destination image
 * \param batch_size Number of images in input
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(const float3 *input,
				   float3 *output,
				   const uint 	batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream = 0);

void shift_corners(const float*		input,
				   float*			output,
				   const uint 	batch_size,
				   const uint		size_x,
				   const uint		size_y,
				   cudaStream_t	stream = 0);

void shift_corners(const cuComplex *input,
				   cuComplex *output,
				   const uint 	batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream = 0);