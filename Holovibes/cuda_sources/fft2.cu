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
#include "hardware_limits.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "transforms.cuh"
#include "preprocessing.cuh"
#include "tools_compute.cuh"
#include "tools.cuh"

enum mode
{
	APPLY_PHASE_FORWARD,
	APPLY_PHASE_INVERSE
};

__global__ static void kernel_fft2_dc(	const complex	*input,
										complex			*output,
										const ushort	width,
										const uint		frame_res,
										const bool		mode)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;;
	if (id < frame_res)
	{
		uint idx = id / width;
		uint idy = id % width;
		complex product;
		if (mode == APPLY_PHASE_FORWARD)
			product = make_cuComplex(cosf(M_PI * (idx + idy)), sinf(M_PI * (idx + idy)));
		else if (mode == APPLY_PHASE_INVERSE)
			product = make_cuComplex(cosf(-M_PI * (idx + idy)), sinf(-M_PI * (idx + idy)));
		output[id] = cuCmulf(input[id], product);
	}
}

void fft_2_dc(	const complex	*input,
				complex			*output,
				const ushort	width,
				const uint		frame_res,
				const uint		p,
				const bool		mode,
				cudaStream_t	stream)
{
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(frame_res, threads);

	kernel_fft2_dc << <blocks, threads, 0, stream >> >(input + frame_res * p, output + frame_res * p, width, frame_res, mode);
	cudaStreamSynchronize(stream);
}

void fft2_lens(	complex					*lens,
				const FrameDescriptor&	fd,
				const float				lambda,
				const float				z,
				cudaStream_t			stream)
{
	uint threads_2d = get_max_threads_2d();
	dim3 lthreads(threads_2d, threads_2d);
	dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);
	//float test = fd.pixel_size;
	kernel_spectral_lens << <lblocks, lthreads, 0, stream >> >(lens, fd, lambda, z);
}

void fft_2(	complex				*input,
			const complex		*lens,
			const cufftHandle	plan1d,
			const cufftHandle	plan2d,
			const uint			frame_resolution,
			const uint			nframes,
			const uint			p,
			const uint			q,
			cudaStream_t		stream)
{
	const uint		n_frame_resolution = frame_resolution * nframes;
	uint			threads = THREADS_128;
	uint			blocks = map_blocks_to_problem(frame_resolution, threads);

	cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);

	cudaStreamSynchronize(stream);

	complex* pframe = input + frame_resolution * p;

	cufftExecC2C(plan2d, pframe, pframe, CUFFT_FORWARD);

	kernel_apply_lens << <blocks, threads, 0, stream >> >(pframe, frame_resolution, lens, frame_resolution);

	cudaStreamSynchronize(stream);


	cufftExecC2C(plan2d, pframe, pframe, CUFFT_INVERSE);

	kernel_complex_divide << <blocks, threads, 0, stream >> >(pframe, frame_resolution, static_cast<float>(n_frame_resolution));

	if (p != q)
	{
		complex* qframe = input + frame_resolution * q;
		cufftExecC2C(plan2d, qframe, qframe, CUFFT_FORWARD);
		kernel_apply_lens << <blocks, threads, 0, stream >> >(qframe, frame_resolution, lens, frame_resolution);
		cufftExecC2C(plan2d, qframe, qframe, CUFFT_INVERSE);
		kernel_complex_divide << <blocks, threads, 0, stream >> >(qframe, frame_resolution, static_cast<float>(n_frame_resolution));
	}
	cudaStreamSynchronize(stream);
}