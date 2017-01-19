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
			uint				stft_level,
			uint				p,
			uint				q,
			uint				frame_size,
			bool				stft_activated,
			cudaStream_t		stream)
{
	//uint threads = 128;
	//uint blocks = map_blocks_to_problem(frame_size, threads);

	// FFT 1D
	if (stft_activated)
		cufftExecC2C(plan1d, gpu_queue, stft_buf, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);
	uint complex_frame_size = sizeof(complex)* frame_size;
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

__global__	void	stft_view_yz(	const complex	*input,
									ushort			*output,
									const uint		x0,
									const uint		frame_size,
									const uint		width,
									const uint		height,
									const uint		depth)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < frame_size)
	{
		const uint index_y = id * width;
		complex pixel = input[x0 + index_y / height * depth + index_y % height];
		//float res = hypotf(pixel.x, pixel.y);
		output[id] = static_cast<ushort>(pixel.x);
	}
}

__global__	void	stft_view_xz(	const complex	*input,
									ushort			*output,
									const uint		y0,
									const uint		frame_size,
									const uint		width)
{
	const uint index_x = blockIdx.x * blockDim.x + threadIdx.x;

	if (index_x < frame_size)
	{
		complex pixel = input[(y0 * width) + (index_x / width) * frame_size + index_x % width];
		//float res = hypotf(pixel.x, pixel.y);
		output[index_x] = static_cast<ushort>(pixel.x);
	}
}

void	stft_view_begin(complex	*input,
						ushort	*outputxz,
						ushort	*outputyz,
						uint	x0,
						uint	y0,
						uint	frame_size,
						uint	width,
						uint	height,
						uint	depth)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_size, threads);

	stft_view_xz<<<blocks, threads, 0, 0>>>(input, outputxz, y0, frame_size, width);
	stft_view_yz<<<blocks, threads, 0, 0>>>(input, outputyz, x0, frame_size, width, height, depth);
}
