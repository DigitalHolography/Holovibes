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
#include "Common.cuh"

#include <cassert>

using holovibes::ImgType;
using holovibes::Queue;
using holovibes::ComputeDescriptor;

// Short-Time Fourier Transform
void stft(Queue				*gpu_stft_queue,
		cuComplex			*gpu_stft_buffer,
		const cufftHandle	plan1d)
{
	// FFT 1D
	cufftSafeCall(cufftExecC2C(plan1d, static_cast<cuComplex*>(gpu_stft_queue->get_buffer()), gpu_stft_buffer, CUFFT_FORWARD));

	// No sync needed since all the kernels are executed on stream 0
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
							const uint			img_type,
							const uint			nSize)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height * nSize)
	{
		float sum = 0;
		for (int x = xmin; x <= xmax; ++x)
		{
			float pixel_float = 0;
			cuComplex pixel = input[x + (id / nSize) * width + (id % nSize) * frame_size];
			if (img_type == ImgType::Modulus || img_type == ImgType::PhaseIncrease || img_type == ImgType::Composite)
				pixel_float = hypotf(pixel.x, pixel.y);
			else if (img_type == ImgType::SquaredModulus)
			{
				pixel_float = hypotf(pixel.x, pixel.y);
				pixel_float *= pixel_float;
			}
			else if (img_type == ImgType::Argument)
				pixel_float = (atanf(pixel.y / pixel.x) + M_PI_2);
			sum += pixel_float;
		}
		output_yz[id] = sum / static_cast<float>(xmax - xmin + 1);
	}
	/* ********** */
	if (id < width * nSize)
	{
		float sum = 0;
		for (int y = ymin; y <= ymax; ++y)
		{
			float pixel_float = 0;
			cuComplex pixel = input[(y * width) + (id / width) * frame_size + id % width];
			if (img_type == ImgType::Modulus || img_type == ImgType::PhaseIncrease || img_type == ImgType::Composite)
				pixel_float = hypotf(pixel.x, pixel.y);
			else if (img_type == ImgType::SquaredModulus)
			{
				pixel_float = hypotf(pixel.x, pixel.y);
				pixel_float *= pixel_float;
			}
			else if (img_type == ImgType::Argument)
				pixel_float = (atanf(pixel.y / pixel.x) + M_PI_2);
			sum += pixel_float;
		}
		output_xz[id] = sum / static_cast<float>(ymax - ymin + 1);
	}
}

void stft_view_begin(const cuComplex	*input,
					float				*output_xz,
					float				*output_yz,
					const ushort		xmin,
					const ushort		ymin,
					const ushort		xmax,
					const ushort		ymax,
					const ushort		width,
					const ushort		height,
					const uint			viewmode,
					const ushort		nSize,
					const uint			acc_level_xz,
					const uint			acc_level_yz,
					const uint			img_type,
					cudaStream_t		stream)
{
	const uint frame_size = width * height;
	const uint output_size = std::max(width, height) * nSize;
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(output_size, threads);

	fill_32bit_slices <<<blocks, threads, 0, stream>>>(
		input,
		output_xz,
		output_yz,
		xmin, ymin, xmax, ymax,
		frame_size,
		output_size,
		width, height,
		acc_level_xz, acc_level_yz,
		img_type,
		nSize);

	cudaCheckError();
}
