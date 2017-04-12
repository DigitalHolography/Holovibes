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
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"

__global__ static void kernel_stft_moment(	complex			*input,
											complex			*output,
											const uint		frame_res,
											ushort			pmin,
											const ushort	pmax)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id] = make_cuComplex(0.f, 0.f);
		while (pmin <= pmax)
		{
			complex *current_pframe = input + (frame_res * pmin);
			output[id].x += current_pframe[id].x;
			output[id].y += current_pframe[id].y;
			++pmin;
		}
	}
}

void stft_moment(	complex			*input, 
					complex			*output,
					const uint		frame_res,
					ushort			pmin,
					const ushort	pmax)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_stft_moment << <blocks, threads, 0, 0 >> > (input, output, frame_res, pmin, pmax);
}

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
	cudaMemcpy(	input,
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

__global__	static void	kernel_stft_view(	const complex	*input,
											float			*output_xz,
											float			*output_yz,
											const uint		start_x,
											const uint		start_y,
											const uint		frame_size,
											const uint		output_size,
											const uint		width,
											const uint		height,
											const uint		depth,
											const uint		acc_level_xz,
											const uint		acc_level_yz)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < output_size)
	{
		complex pixel = make_cuComplex(0, 0);
		int i = -1;
		uint img_acc_level = acc_level_yz;
		while (++i < img_acc_level)
		{
			pixel = cuCaddf(pixel, input[start_x + i + id * width]);
		}
		output_yz[id] = hypotf(pixel.x, pixel.y) / 5.f;
		i = -1;
		pixel = make_cuComplex(0, 0);
		img_acc_level = acc_level_xz;
		while (++i < img_acc_level)
		{
			pixel = cuCaddf(pixel, input[((start_y + i) * width) + (id / width) * frame_size + id % width]);
		}
		output_xz[id] = hypotf(pixel.x, pixel.y) / 5.f;
	}
}

void	stft_view_begin(const complex	*input,
						float			*outputxz,
						float			*outputyz,
						const uint		start_x,
						const uint		start_y,
						const uint		width,
						const uint		height,
						const uint		depth,
						const uint		acc_level_xz,
						const uint		acc_level_yz)
{
	const uint frame_size = width * height;
	const uint output_size = width * depth;
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(output_size, threads); 

	kernel_stft_view << <blocks, threads, 0, 0 >> >(input,
													outputxz,
													outputyz,
													start_x,
													start_y,
													frame_size,
													output_size,
													width,
													height,
													depth,
													acc_level_xz,
													acc_level_yz);
}