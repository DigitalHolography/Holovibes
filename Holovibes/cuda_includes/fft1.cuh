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

#include "Common.cuh"

/*! \brief Find the right threads and block to call quadratic lens
* with and call it
*/
void fft1_lens(cuComplex*	lens,
	const uint 				lens_side_size,
	const uint 				frame_height,
	const uint 				frame_width,
	const float				lambda,
	const float				z,
	const float				pixel_size,
	cudaStream_t			stream = 0);

/*! \brief Apply a lens and call an fft1 on the image
*
* \param lens the lens that will be applied to the image
* \param plan the first paramater of cufftExecC2C that will be called
* on the image
*/
void fft_1(cuComplex*			input,
		cuComplex* 				output,
		const uint 				batch_size,
		const cuComplex*		lens,
		const cufftHandle		plan2D,
		const uint				frame_resolution,
		cudaStream_t			stream = 0);