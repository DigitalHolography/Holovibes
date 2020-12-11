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

#include "fft2.cuh"
#include "transforms.cuh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"

#include <cufftXt.h>


using camera::FrameDescriptor;

enum mode
{
	APPLY_PHASE_FORWARD,
	APPLY_PHASE_INVERSE
};

__global__
static void kernel_fft2_dc(const cuComplex* const 	input,
						   cuComplex* const			output,
						   const ushort				width,
						   const uint				frame_res,
						   const uint 				batch_size,
						   const bool				mode)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < frame_res)
	{
		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * frame_res;

			const float	pi_pxl = M_PI * (index / width + index % width);
			if (mode == APPLY_PHASE_FORWARD)
				output[batch_index] = cuCmulf(input[batch_index], make_cuComplex(cosf(pi_pxl), sinf(pi_pxl)));
			else if (mode == APPLY_PHASE_INVERSE)
				output[batch_index] = cuCmulf(input[batch_index], make_cuComplex(cosf(-pi_pxl), sinf(-pi_pxl)));
		}
	}
}

static void fft_2_dc(const ushort	width,
				const uint		frame_res,
				cuComplex		*pframe,
				const bool		mode,
				const uint 		batch_size,
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_fft2_dc << <blocks, threads, 0, stream >> >(pframe, pframe, width, frame_res, batch_size, mode);
	cudaCheckError();
}

void fft2_lens(cuComplex*	lens,
	const uint 				lens_side_size,
	const uint 				frame_height,
	const uint 				frame_width,
	const float				lambda,
	const float				z,
	const float				pixel_size,
	cudaStream_t			stream)
{
	const uint threads_2d = get_max_threads_2d();
	const dim3 lthreads(threads_2d, threads_2d);
	const dim3 lblocks(lens_side_size / threads_2d, lens_side_size / threads_2d);

	cuComplex* square_lens;
	// In anamorphic mode, the lens is initally a square, it's then cropped to be the same dimension as the frame
	if (frame_height != frame_width)
		cudaXMalloc(&square_lens, lens_side_size * lens_side_size * sizeof(cuComplex));
	else
		square_lens = lens;

	kernel_spectral_lens<<<lblocks, lthreads, 0, stream>>>(square_lens, lens_side_size, lambda, z, pixel_size);
	cudaCheckError();

	if (frame_height != frame_width)
	{
		cudaXMemcpy(lens, square_lens + ((lens_side_size - frame_height) / 2) * frame_width, frame_width * frame_height * sizeof(cuComplex));
		cudaXFree(square_lens);
	}
}

void fft_2(cuComplex			*input,
		cuComplex 				*output,
		const uint 				batch_size,
		const cuComplex			*lens,
		const cufftHandle		plan2d,
		const FrameDescriptor&	fd,
		cudaStream_t			stream)
{
	const uint frame_resolution = fd.frame_res();
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_resolution, threads);

	fft_2_dc(fd.width, frame_resolution, input, 0, batch_size, stream);

	cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

	kernel_apply_lens<<<blocks, threads, 0, stream>>>(input, output, batch_size, frame_resolution, lens, frame_resolution);
	cudaCheckError();

	cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));

	fft_2_dc(fd.width, frame_resolution, input, 1, batch_size, stream);

	kernel_complex_divide<<<blocks, threads, 0, stream>>>(input, frame_resolution, static_cast<float>(frame_resolution), batch_size);
	cudaCheckError();
}