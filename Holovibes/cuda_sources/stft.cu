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

// Short-Time Fourier Transform
void stft(cuComplex			*input,
		cuComplex			*gpu_queue,
		cuComplex			*stft_buf,
		const cufftHandle	plan1d,
		const uint			tft_level,
		const uint			p,
		const uint			q,
		const uint			frame_size,
		const bool			stft_activated,
		cudaStream_t		stream)
{
	const uint complex_frame_size = sizeof(cuComplex) * frame_size;

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

__global__
static void kernel_composite(cuComplex	*input,
							cuComplex		*output,
							const uint		frame_res,
							ushort			*p_array)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id] = make_cuComplex(0.f, 0.f);
		ushort p = p_array[id % 3];
		cuComplex *current_pframe = input + (frame_res * p);
		output[id].x += hypotf(current_pframe[id / 3].x, current_pframe[id / 3].y);
	}
}
void composite(cuComplex	*input,
			cuComplex		*output,
			const uint		frame_res,
			ushort			*p_array)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_composite << <blocks, threads, 0, 0 >> > (input, output, frame_res, p_array);
}
#pragma region moment
__global__
static void kernel_stft_moment(cuComplex	*input,
							cuComplex		*output,
							const uint		frame_res,
							ushort			pmin,
							ushort			pmax,
							const uint		nsamples)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	const uint nb_slices_p = pmax - pmin + 1;
	if (id < frame_res)
	{
		output[id] = make_cuComplex(0.f, 0.f);
		while (pmin <= pmax)
		{
			cuComplex *current_pframe = input + (frame_res * pmin);
			output[id].x += hypotf(current_pframe[id].x, current_pframe[id].y);
			++pmin;
		}
		output[id].x /= nb_slices_p;
	}
}

void stft_moment(cuComplex		*input,
				cuComplex		*output,
				const uint		frame_res,
				ushort			pmin,
				const ushort	pmax,
				const uint		nsamples)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_stft_moment << <blocks, threads, 0, 0 >> > (input, output, frame_res, pmin, pmax, nsamples);
}
#pragma endregion


__global__
static void	fill_64bit_slices(const cuComplex	*input,
							cuComplex			*output_xz,
							cuComplex			*output_yz,
							const uint			start_x,
							const uint			start_y,
							const uint			frame_size,
							const uint			output_size,
							const uint			width,
							const uint			height,
							const uint			acc_level_xz,
							const uint			acc_level_yz)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < output_size)
	{
		output_xz[id] = input[start_x * width + (id / width) * frame_size + id % width];
		output_yz[id] = input[start_x + id * width];
	}
}

__global__
static void	fill_32bit_slices(const cuComplex	*input,
							float				*output_xz,
							float				*output_yz,
							const uint			xmin,
							const uint			ymin,
							const uint			xmax,
							const uint			ymax,
							const uint			frame_size,
							const uint			output_size,
							const uint			width,
							const uint			height,
							const uint			acc_level_xz,
							const uint			acc_level_yz,
							const uint			img_type)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < output_size)
	{
		cuComplex pixel = make_cuComplex(0, 0);
		for (int i = xmin; i <= xmax; ++i)
			pixel = cuCaddf(pixel, input[i + id * width]);
		if (img_type == ImgType::Modulus || img_type == ImgType::PhaseIncrease || img_type == ImgType::Composite)
			output_yz[id] = hypotf(pixel.x, pixel.y);
		else if (img_type == ImgType::SquaredModulus)
		{
			output_yz[id] = hypotf(pixel.x, pixel.y);
			output_yz[id] *= output_yz[id];
		}
		else if (img_type == ImgType::Argument)
			output_yz[id] = (atanf(pixel.y / pixel.x) + M_PI_2);
		output_yz[id] /= static_cast<float>(xmax - xmin + 1);
		/* ********** */
		pixel = make_cuComplex(0, 0);
		for (int i = ymin; i <= ymax; ++i)
			pixel = cuCaddf(pixel, input[(i * width) + (id / width) * frame_size + id % width]);
		if (img_type == ImgType::Modulus || img_type == ImgType::PhaseIncrease || img_type == ImgType::Composite)
			output_xz[id] = hypotf(pixel.x, pixel.y);
		else if (img_type == ImgType::SquaredModulus)
		{
			output_xz[id] = hypotf(pixel.x, pixel.y);
			output_xz[id] *= output_xz[id];
		}
		else if (img_type == ImgType::Argument)
			output_xz[id] = (atanf(pixel.y / pixel.x) + M_PI_2);
		output_xz[id] /= static_cast<float>(ymax - ymin + 1);
	}
}

void stft_view_begin(const cuComplex	*input,
					void				*output_xz,
					void				*output_yz,
					const ushort		xmin,
					const ushort		ymin,
					const ushort		xmax,
					const ushort		ymax,
					const ushort		width,
					const ushort		height,
					const uint			viewmode,
					const ushort		nsamples,
					const uint			acc_level_xz,
					const uint			acc_level_yz,
					const uint			img_type)
{
	const uint frame_size = width * height;
	const uint output_size = width * nsamples;
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(output_size, threads); 

	if (viewmode == ImgType::Complex)
		fill_64bit_slices << <blocks, threads, 0, 0 >> >(
			input,
			reinterpret_cast<cuComplex *>(output_xz),
			reinterpret_cast<cuComplex *>(output_yz),
			xmin, ymin,
			frame_size,
			output_size,
			width, height,
			acc_level_xz, acc_level_yz);
	else
		fill_32bit_slices <<<blocks, threads, 0, 0>>>(
			input,
			reinterpret_cast<float *>(output_xz),
			reinterpret_cast<float *>(output_yz),
			xmin, ymin, xmax, ymax,
			frame_size,
			output_size,
			width, height,
			acc_level_xz, acc_level_yz,
			img_type);
}
