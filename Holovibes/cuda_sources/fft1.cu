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
#include "hardware_limits.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"
#include "preprocessing.cuh"
#include "transforms.cuh"

void fft1_lens(	complex*				lens,
				const FrameDescriptor&	fd,
				const float				lambda,
				const float				z,
				cudaStream_t			stream)
{
  uint threads = 128;
  uint blocks = map_blocks_to_problem(fd.frame_res(), threads);

  kernel_quadratic_lens << <blocks, threads, 0, stream >> >(lens, fd, lambda, z);
}

void fft_1(	complex				*input,
			const complex*		lens,
			const cufftHandle	plan1D,
			const cufftHandle	plan2D,
			const uint			frame_resolution,
			const uint			nframes,
			const uint			p,
			const uint			q,
			cudaStream_t		stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);
	
	complex* pframe = input + frame_resolution * p;

	cufftExecC2C(plan1D, input, input, CUFFT_FORWARD);

	// Apply lens on multiple frames.
	kernel_apply_lens <<<blocks, threads, 0, stream>>>(pframe, frame_resolution, lens, frame_resolution);
	cudaStreamSynchronize(stream);
	// FFT
    cufftExecC2C(plan2D, pframe, pframe, CUFFT_FORWARD);
	if (p != q)
	{
		complex *qframe = input + frame_resolution * q;
		kernel_apply_lens <<<blocks, threads, 0, stream>>>(qframe, frame_resolution, lens, frame_resolution);
		cufftExecC2C(plan2D, qframe, qframe, CUFFT_FORWARD);
	}

	cudaStreamSynchronize(stream);
}
/*
void fft_1(	complex				*input,
			const complex		*lens,
			const cufftHandle	plan,
			const uint			frame_resolution,
			const uint			nframes,
			cudaStream_t		stream)
{
  const uint n_frame_resolution = frame_resolution * nframes;

  uint threads = get_max_threads_1d();
  uint blocks = map_blocks_to_problem(frame_resolution, threads);

  // Apply lens on multiple frames.
  kernel_apply_lens << <blocks, threads, 0, stream >> >(input, n_frame_resolution, lens, frame_resolution);

  cudaStreamSynchronize(stream);
  // FFT
  cufftResult res = cufftExecC2C(plan, input, input, CUFFT_FORWARD);
  
  cudaStreamSynchronize(stream);
}*/
