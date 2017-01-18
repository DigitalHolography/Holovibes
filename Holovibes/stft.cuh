/*! \file 
 *
 * Functions that will be used to compute the stft.
 */
#pragma once

#include "cuda_shared.cuh"

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
void stft(	complex				*input,
			complex				*gpu_queue,
			complex				*stft_buf,
			const cufftHandle	plan1d,
			uint				stft_level,
			uint				p,
			uint				q,
			uint				frame_size,
			bool				stft_activated,
			cudaStream_t		stream = 0);

void	stft_view_begin(	complex	*input,
							ushort	*output,
							uint	x0,
							uint	y0,
							uint	frame_size,
							uint	width,
							uint	height,
							uint	depth);
