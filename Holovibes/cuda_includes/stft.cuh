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

/*! \brief Function handling the stft algorithm which steps are \n
 * 1 : Aplly lens on the input queue \n
 * 2 : Do a fft2d (fresnel transform) on the input queue \n
 * 3 : Take the ROI (i.e. 512x512px) and store bursting way on a complex queue (stft_buf) \n
 * 4 : Do nsamples fft1d on the complex queue (stft_dup_buf) \n
 * This complex queue need to be reconstruct in order to get image
 *
 * \param stft_buf the buffer which will be exploded
 * \param stft_dup_buf the buffer that will receive the plan1d transforms
 * \param r The rectangle selected for the stft to be done on it
 *
 * \verbatim
 *
 *    [a1][b1]   [a2][b2]   [a3][b3]
 *    [c1][d1]   [c2][d2]   [c3][d3]
 *     img 1       img 2      img 3
 *
 *         **   Bursting  **
 *  [a1][a2][a3][b1][b2][b3][c1][...]
 *
 *         **   plan1d    **
 *      [a123][b123][c123][d123]
 *
 *         ** Reconstruct **
 *           [a123][b123]
 *           [c123][d123]
 *
 *\endverbatim
 */
void stft(cuComplex			*input,
		cuComplex			*gpu_queue,
		cuComplex			*stft_buf,
		const cufftHandle	plan1d,
		const uint			stft_level,
		const uint			p,
		const uint			q,
		const uint			frame_size,
		const bool			stft_activated,
		cudaStream_t		stream = 0);

/// Computes the average frame for pmin <= p <= pmax
void stft_moment(cuComplex		*input,
				cuComplex		*output,
				const uint		frame_res,
				ushort			pmin,
				const ushort	pmax,
				const uint		nsamples);
/// Computes 3 different p slices and put them in each color
void composite(cuComplex	*input,
			cuComplex		*output,
			const uint		frame_res,
			ushort			*p_array);

void stft_block_to_float(const cufftComplex	*input,
						float				*output,
						const uint			size);

void stft_view_begin(const cuComplex	*input,
					void				*output_xz,
					void				*output_yz,
					const ushort		xmin,
					const ushort		ymin,
					const ushort		xmax,
					const ushort		ymax,
					const ushort		width,
					const ushort		height,
					const uint			viewmode,
					const ushort		nsamples,
					const uint			acc_level_xz,
					const uint			acc_level_yz,
					const uint			img_type);
