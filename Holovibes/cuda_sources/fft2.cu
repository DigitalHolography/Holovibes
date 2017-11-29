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
#include "preprocessing.cuh"
#include "transforms.cuh"
#include "tools_compute.cuh"

using camera::FrameDescriptor;

enum mode
{
	APPLY_PHASE_FORWARD,
	APPLY_PHASE_INVERSE
};

__global__
static void kernel_fft2_dc(const cuComplex	*input,
						cuComplex			*output,
						const ushort		width,
						const uint			frame_res,
						const bool			mode)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;;
	if (id < frame_res)
	{
		const float	pi_pxl = M_PI * (id / width + id % width);
		if (mode == APPLY_PHASE_FORWARD)
			output[id] = cuCmulf(input[id], make_cuComplex(cosf(pi_pxl), sinf(pi_pxl)));
		else if (mode == APPLY_PHASE_INVERSE)
			output[id] = cuCmulf(input[id], make_cuComplex(cosf(-pi_pxl), sinf(-pi_pxl)));
	}
}

void fft_2_dc(	const ushort	width,
				const uint		frame_res,
				cuComplex		*pframe,
				const bool		mode,
				cudaStream_t	stream)
{
	const uint	threads = get_max_threads_1d();
	const uint	blocks = map_blocks_to_problem(frame_res, threads);

	kernel_fft2_dc << <blocks, threads, 0, stream >> >(pframe, pframe, width, frame_res, mode);
}

void fft2_lens(cuComplex			*lens,
			const FrameDescriptor&	fd,
			const float				lambda,
			const float				z,
			const float				pixel_size,
			cudaStream_t			stream)
{
	uint threads_2d = get_max_threads_2d();
	dim3 lthreads(threads_2d, threads_2d);
	dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);

	kernel_spectral_lens << <lblocks, lthreads, 0, stream >> >(lens, fd, lambda, z, pixel_size);
}

void fft_2(cuComplex			*input,
		const cuComplex			*lens,
		const cufftHandle		plan2d,
		const FrameDescriptor&	fd,
		cudaStream_t			stream)
{
	const uint	frame_resolution = fd.frame_res();
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_resolution, threads);

	cudaStreamSynchronize(stream);

	fft_2_dc(fd.width, frame_resolution, input, 0, stream);

	cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);

	kernel_apply_lens << <blocks, threads, 0, stream >> >(input, frame_resolution, lens, frame_resolution);

	cudaStreamSynchronize(stream);


	cufftExecC2C(plan2d, input, input, CUFFT_INVERSE);

	fft_2_dc(fd.width, frame_resolution, input, 1, stream);

	kernel_complex_divide << <blocks, threads, 0, stream >> >(input, frame_resolution, static_cast<float>(frame_resolution));

	cudaStreamSynchronize(stream);
}
