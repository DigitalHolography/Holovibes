/*! \file
 *
 * Functions that run the fft2 and the computation of the fft2 lens.
 */
#pragma once

# include "cuda_shared.cuh"

/*! \brief Find the right threads and block to call spectral lens
* with and call it
*/
void fft2_lens(	complex							*lens,
				const camera::FrameDescriptor&	fd,
				const float						lambda,
				const float						z,
				cudaStream_t					stream = 0);

/*! \brief takes input complex buffer and computes a p frame that is stored
 * at output pointer. The output pointer can be another complex buffer or the
 * same as input buffer.
 */
void fft_2(	complex				*input,
			const complex		*lens,
			const cufftHandle	plan1d,
			const cufftHandle	plan2d,
			const uint			frame_resolution,
			const uint			nframes,
			const uint			p,
			const uint			q,
			cudaStream_t		stream = 0);