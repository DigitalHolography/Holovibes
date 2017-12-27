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
#include "remove_jitter.hh"

using holovibes::ImgType;
using holovibes::Queue;
using holovibes::ComputeDescriptor;
using holovibes::compute::RemoveJitter;

struct Zone
{
	Zone(RectFd z)
		: x(z.topLeft().x())
		, y(z.topLeft().y())
		, x2(z.bottomRight().x())
		, y2(z.bottomRight().y())
		, area(z.area())
	{}

	int x;
	int y;
	int x2;
	int y2;
	int area;
};

__global__
static void kernel_zone_copy(cuComplex		*src,
							cuComplex		*dst,
							const uint		nSize,
							const uint		width,
							const uint		height,
							Zone			zone)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < zone.area * nSize)
	{
		const uint frame_res = width * height;
		const uint zone_width = zone.x2 - zone.x;
		const uint line = (id % zone.area) / zone_width;
		const uint column = (id % zone.area) % zone_width;
		const uint depth = id / zone.area;

		const uint src_pos = (depth * frame_res) + ((zone.y + line) * width) + (zone.x + column);
		dst[id] = src[src_pos];
	}
}

static void zone_copy(cuComplex		*src,
					cuComplex		*dst,
					const uint		nSize,
					const uint		width,
					const uint		height,
					RectFd			cropped_zone,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	Zone zone(cropped_zone);
	const uint blocks = map_blocks_to_problem(zone.area * nSize, threads);

	kernel_zone_copy<<<blocks, threads, 0, stream>>>(src, dst, nSize, width, height, zone);
	cudaCheckError();
}

__global__
static void kernel_zone_uncopy(cuComplex	*src,
							cuComplex		*dst,
							const uint		nSize,
							const uint		width,
							const uint		height,
							Zone			zone)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < zone.area * nSize)
	{
		const uint frame_res = width * height;
		const uint zone_width = zone.x2 - zone.x;
		const uint line = (id % zone.area) / zone_width;
		const uint column = (id % zone.area) % zone_width;
		const uint depth = id / zone.area;

		const uint dst_pos = (depth * frame_res) + ((zone.y + line) * width) + (zone.x + column);
		dst[dst_pos] = src[id];
	}
}

static void zone_uncopy(cuComplex		*src,
					cuComplex		*dst,
					const uint		nSize,
					const uint		width,
					const uint		height,
					RectFd			cropped_zone)
{
	const uint threads = get_max_threads_1d();
	Zone zone(cropped_zone);
	const uint blocks = map_blocks_to_problem(zone.area * nSize, threads);

	kernel_zone_uncopy<<<blocks, threads, 0, 0>>> (src, dst, nSize, width, height, zone);
	cudaCheckError();
}

/* \brief Copy a circular queue into a contiguous buffer using zone_copy
 */
static void unwrap_circular_queue(Queue		*queue,
								cuComplex	*dst,
								RectFd		cropped_zone,
								cudaStream_t	stream = 0)
{
	// Most variables here will be optimized away,
	// they just make the process clearer and debugging easier

	const auto queue_buffer = static_cast<cuComplex*>(queue->get_buffer());
	const uint index_first_element = queue->get_start_index();
	const uint queue_size = queue->get_max_elts();
	const uint width = queue->get_frame_desc().width;
	const uint height = queue->get_frame_desc().height;
	const uint frame_size = queue->get_frame_desc().frame_res();

	const uint first_block_size = queue_size - index_first_element;
	const uint second_block_size = queue_size - first_block_size;
	const auto first_block_from = queue_buffer + index_first_element * frame_size;
	// reminder : &a[b] == (a + b)
	const auto second_block_from = queue_buffer;
	const auto first_block_to = dst;
	const auto second_block_to = dst + first_block_size * cropped_zone.area();

	if (first_block_size)
		zone_copy(first_block_from, first_block_to, first_block_size, width, height, cropped_zone, stream);
	if (second_block_size)
		zone_copy(second_block_from, second_block_to, second_block_size, width, height, cropped_zone, stream);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

// Short-Time Fourier Transform
void stft(cuComplex			*input,
		Queue				*gpu_queue,
		cuComplex			*stft_buf,
		const cufftHandle	plan1d,
		const uint			q,
		const uint			width,
		const uint			height,
		const bool			stft_activated,
		const ComputeDescriptor &cd,
		cuComplex			*cropped_stft_buf,
		cudaStream_t		stream)
{
	const uint tft_level = cd.nSize;
	const uint p = cd.pindex;
	const uint nSize = cd.nSize;
	bool cropped_stft = cd.croped_stft;
	const RectFd cropped_zone = cd.getZoomedZone();

	const int frame_size = width * height;
	const uint complex_frame_size = sizeof(cuComplex) * frame_size;

	// FFT 1D
	if (stft_activated)
	{
		if (cropped_stft)
		{
			unwrap_circular_queue(gpu_queue, cropped_stft_buf, cropped_zone, stream);

			RemoveJitter jitter(cropped_stft_buf, cropped_zone, cd);
			jitter.run();

			cufftExecC2C(plan1d, cropped_stft_buf, cropped_stft_buf, CUFFT_FORWARD);
			zone_uncopy(cropped_stft_buf, stft_buf, nSize, width, height, cropped_zone);
		}
		else
			cufftExecC2C(plan1d, static_cast<cuComplex*>(gpu_queue->get_buffer()), stft_buf, CUFFT_FORWARD);
	}
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

#pragma region moment
__global__
static void kernel_stft_moment(cuComplex	*input,
							cuComplex		*output,
							const uint		frame_res,
							ushort			pmin,
							ushort			pmax)
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
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_stft_moment << <blocks, threads, 0, stream >> > (input, output, frame_res, pmin, pmax);
	cudaCheckError();
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
					void				*output_xz,
					void				*output_yz,
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

	if (viewmode == ImgType::Complex)
		fill_64bit_slices << <blocks, threads, 0, stream >> >(
			input,
			static_cast<cuComplex *>(output_xz),
			static_cast<cuComplex *>(output_yz),
			xmin, ymin,
			frame_size,
			output_size,
			width, height,
			acc_level_xz, acc_level_yz);
	else
		fill_32bit_slices <<<blocks, threads, 0, stream>>>(
			input,
			static_cast<float *>(output_xz),
			static_cast<float *>(output_yz),
			xmin, ymin, xmax, ymax,
			frame_size,
			output_size,
			width, height,
			acc_level_xz, acc_level_yz,
			img_type,
			nSize);
	cudaCheckError();
}
