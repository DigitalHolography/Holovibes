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

# include "cuda_shared.cuh"

void fft_2_dc(	const complex	*input,
				complex			*output,
				const ushort	width,
				const uint		frame_res,
				const uint		p,
				const bool		mode,
				cudaStream_t	stream);

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