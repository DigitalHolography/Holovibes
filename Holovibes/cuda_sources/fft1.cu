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

#include "fft1.cuh"
#include "preprocessing.cuh"
#include "transforms.cuh"

void fft1_lens(cuComplex*			lens,
			const FrameDescriptor&	fd,
			const float				lambda,
			const float				z,
			const float				pixel_size,
			cudaStream_t			stream)
{
  uint threads = 128;
  uint blocks = map_blocks_to_problem(fd.frame_res(), threads);

  kernel_quadratic_lens << <blocks, threads, 0, stream >> >(lens, fd, lambda, z, pixel_size);
}

void fft_1(cuComplex*			input,
		const cuComplex*		lens,
		const cufftHandle		plan1D,
		const cufftHandle		plan2D,
		const uint				frame_resolution,
		const uint				nframes,
		const uint				p,
		const uint				q,
		cudaStream_t			stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);
	
	cuComplex* pframe = input + frame_resolution * p;

	cufftExecC2C(plan1D, input, input, CUFFT_FORWARD);

	// Apply lens on multiple frames.
	kernel_apply_lens <<<blocks, threads, 0, stream>>>(pframe, frame_resolution, lens, frame_resolution);
	cudaStreamSynchronize(stream);
	// FFT
    cufftExecC2C(plan2D, pframe, pframe, CUFFT_FORWARD);
	if (p != q)
	{
		cuComplex *qframe = input + frame_resolution * q;
		kernel_apply_lens <<<blocks, threads, 0, stream>>>(qframe, frame_resolution, lens, frame_resolution);
		cufftExecC2C(plan2D, qframe, qframe, CUFFT_FORWARD);
	}

	cudaStreamSynchronize(stream);
}
