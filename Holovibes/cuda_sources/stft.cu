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

#include "stft.cuh"
#include "hardware_limits.hh"
#include "geometry.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"
#include "geometry.hh"

void stft(	complex				*input,
			complex				*gpu_queue,
			complex				*stft_buf,
			const cufftHandle	plan1d,
			const uint			tft_level,
			const uint			p,
			const uint			q,
			const uint			frame_size,
			const bool			stft_activated,
			cudaStream_t		stream)
{
	const uint complex_frame_size = sizeof(complex) * frame_size;
	// FFT 1D
	if (stft_activated)
		cufftExecC2C(plan1d, gpu_queue, stft_buf, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);
	cudaMemcpy(
		input,
		stft_buf + p * frame_size,
		complex_frame_size,
		cudaMemcpyDeviceToDevice);

	if (p != q)
	{
	cudaMemcpy(	input + frame_size,
				stft_buf + q * frame_size,
				complex_frame_size,
				cudaMemcpyDeviceToDevice);
	}
}

__global__	void	kernel_stft_view_xz(const complex	*input,
										float			*output,
										const uint		x0,
										const uint		y0,
										const uint		frame_size,
										const uint		output_size,
										const uint		width,
										const uint		height,
										const uint		depth)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < output_size)
	{
		const complex pixel = input[(y0 * width) + (id / width) * frame_size + id % width];
		output[id] = hypotf(pixel.x, pixel.y);
	}
}

__global__	void	kernel_stft_view_yz(const complex	*input,
										float			*output,
										const uint		x0,
										const uint		y0,
										const uint		frame_size,
										const uint		output_size,
										const uint		width,
										const uint		height,
										const uint		depth)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < output_size)
	{
		const complex pixel = input[x0 + id * width];
		output[id] = hypotf(pixel.x, pixel.y);
	}
}

void	stft_view_begin(const complex	*input,
						float			*outputxz,
						float			*outputyz,
						const uint		x0,
						const uint		y0,
						const uint		width,
						const uint		height,
						const uint		depth)
{
	const uint frame_size = width * height;
	const uint output_size_xz = width * depth;
	const uint output_size_yz = height * depth;
	const uint threads = get_max_threads_1d();
	const uint blocks_xz = map_blocks_to_problem(output_size_xz, threads);
	const uint blocks_yz = map_blocks_to_problem(output_size_yz, threads);
	
	kernel_stft_view_xz << <blocks_xz, threads, 0, 0 >> >(input, outputxz, x0, y0, frame_size, output_size_xz, width, height, depth);
	kernel_stft_view_yz << <blocks_yz, threads, 0, 0 >> >(input, outputyz, x0, y0, frame_size, output_size_yz, width, height, depth);
}