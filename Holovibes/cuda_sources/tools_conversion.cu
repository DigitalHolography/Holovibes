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

#include "tools_conversion.cuh"

__global__
void img8_to_complex(cuComplex		*output,
					const uchar		*input,
					const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		// Image rescaling on 2^16 colors (65535 / 255 = 257)
		const float val = static_cast<float>(input[index] * 257);
		output[index].x = val;
		output[index].y = 0;
	}
}

__global__
void img16_to_complex(cuComplex		*output,
					const ushort	*input,
					const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		float val = static_cast<float>(input[index]);
		output[index].x = val;
		output[index].y = 0;
	}
}

__global__
void float_to_complex(cuComplex	*output,
					const float	*input,
					const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		float val = input[index];
		output[index].x = val;
		output[index].y = 0;
	}
}

/* Kernel function wrapped by complex_to_modulus. */
static __global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = hypotf(input[index].x, input[index].y);
	}
}

void complex_to_modulus(const cuComplex	*input,
						float			*output,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_modulus << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Kernel function wrapped in complex_to_squared_modulus. */
static __global__
void kernel_complex_to_squared_modulus(const cuComplex	*input,
									float				*output,
									const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = hypotf(input[index].x, input[index].y);
		output[index] *= output[index];
	}
}

void complex_to_squared_modulus(const cuComplex	*input,
								float			*output,
								const uint		size,
								cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_squared_modulus << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Kernel function wrapped in complex_to_argument. */
static __global__
void kernel_complex_to_argument(const cuComplex	*input,
								float			*output,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = (atanf(input[index].y / input[index].x) + M_PI_2);
	}
}

void complex_to_argument(const cuComplex	*input,
						float			*output,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_argument << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Find the minimum and the maximum of a floating-point array.
 *
 * The minimum and maximum can't be computed directly, because blocks
 * cannot communicate. Hence we compute local minima and maxima and
 * put them in two arrays.
 *
 * \param Size Number of threads in a block for this kernel.
 * Also, it's the size of min and max.
 * \param min Array of Size floats, which will contain local minima.
 * \param max Array of Size floats, which will contain local maxima.
 */
template <uint Size>
static __global__
void kernel_minmax(const float	*data,
				const size_t	size,
				float			*min,
				float			*max)
{
	__shared__ float local_min[Size];
	__shared__ float local_max[Size];

	const uint index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > size)
		return;
	local_min[threadIdx.x] = data[index];
	local_max[threadIdx.x] = data[index];

	__syncthreads();

	if (threadIdx.x == 0)
	{
		/* Accumulate the results of the neighbors, computing min-max values,
		 * and store them in the first element of local arrays. */
		for (auto i = 1; i < Size; ++i)
		{
			if (local_min[i] < local_min[0])
				local_min[0] = local_min[i];
			if (local_max[i] > local_max[0])
				local_max[0] = local_max[i];
		}
		min[blockIdx.x] = local_min[0];
		max[blockIdx.x] = local_max[0];
	}
}

template <typename T>
static __global__
void kernel_rescale(T				*data,
					const size_t	size,
					const T			min,
					const T			max,
					const T			new_max)
{
	const uint index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index > size)
		return;

	data[index] = (data[index] + fabsf(min)) * new_max / (fabsf(max) + fabsf(min));
}

void rescale_float(const float	*input,
				float			*output,
				const uint		size,
				cudaStream_t	stream)
{
	const uint threads = THREADS_128;
	const uint blocks = map_blocks_to_problem(size, threads);

	// TODO : See if gpu_float_buffer_ could be used directly.
	cudaMemcpy(output, input, sizeof(float) * size, cudaMemcpyDeviceToDevice);

	// Computing minimum and maximum values, in order to rescale properly.
	float* gpu_local_min;
	float* gpu_local_max;
	const uint float_blocks = sizeof(float) * blocks;
	if (cudaMalloc(&gpu_local_min, float_blocks) != cudaSuccess)
		return;
	if (cudaMalloc(&gpu_local_max, float_blocks) != cudaSuccess)
	{
		cudaFree(gpu_local_min);
		return;
	}

	/* We have to hardcode the template parameter, unfortunately.
	 * It must be equal to the number of threads per block. */
	kernel_minmax <128> << <blocks, threads, threads << 1, stream >> > (output, size, gpu_local_min, gpu_local_max);

	float	*cpu_local_min = new float[blocks];
	float	*cpu_local_max = new float[blocks];
	cudaMemcpy(cpu_local_min, gpu_local_min, float_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_local_max, gpu_local_max, float_blocks, cudaMemcpyDeviceToHost);

	const float max_intensity = 65535.f;
	kernel_rescale << <blocks, threads, 0, stream >> >(	output,
														size,
														*(std::min_element(cpu_local_min, cpu_local_min + threads)),
														*(std::max_element(cpu_local_max, cpu_local_max + threads)),
														max_intensity);
	delete[] cpu_local_max;
	delete[] cpu_local_min;
	cudaFree(gpu_local_min);
	cudaFree(gpu_local_max);
}

void rescale_float_unwrap2d(float			*input,
							float			*output,
							float			*cpu_buffer,
							uint			frame_res,
							cudaStream_t	stream)
{
	float		min = 0;
	float		max = 0;
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(frame_res, threads);
	uint float_frame_res = sizeof(float)* frame_res;
	cudaMemcpy(cpu_buffer, input, float_frame_res, cudaMemcpyDeviceToHost);
	auto minmax = std::minmax_element(cpu_buffer, cpu_buffer + frame_res);
	min = *minmax.first;
	max = *minmax.second;

	cudaMemcpy(output, input, float_frame_res, cudaMemcpyDeviceToDevice);

	kernel_normalize_images << < blocks, threads, 0, stream >> > (
		output,
		max,
		min,
		frame_res);
}

__global__
void kernel_rescale_argument(float		*input,
							const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		input[index] *= 65535.0f / M_PI;
	}
}

void rescale_argument(float			*input,
					const uint		frame_res,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_rescale_argument << <blocks, threads, 0, stream >> >(input, frame_res);
}

/*! \brief Kernel function wrapped in endianness_conversion, making
 ** the call easier
 **/
static __global__
void kernel_endianness_conversion(const ushort	*input,
								ushort			*output,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		output[index] = (input[index] << 8) | (input[index] >> 8);
	}
}

void endianness_conversion(const ushort	*input,
						ushort			*output,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_endianness_conversion << <blocks, threads, 0, stream >> >(input, output, size);
}

__global__
static void kernel_composite(cuComplex			*input,
							float				*output,
							const uint			frame_res,
							ushort				pmin_r,
							ushort				pmax_r,
							float				weight_r,
							ushort				pmin_g,
							ushort				pmax_g,
							float				weight_g,
							ushort				pmin_b,
							ushort				pmax_b,
							float				weight_b)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	ushort pmin[] = { pmin_r, pmin_g, pmin_b };
	ushort pmax[] = { pmax_r, pmax_g, pmax_b };
	float weight[] = { weight_r, weight_g, weight_b };
	if (id < frame_res)
	{
		for (int i = 0; i < 3; i++)
		{
			float res = 0;
			for (ushort p = pmin[i]; p <= pmax[i]; p++)
			{
				cuComplex *current_pframe = input + (frame_res * p);
				res += hypotf(current_pframe[id].x, current_pframe[id].y);
			}
			output[id * 3 + i] = res * weight[i] / (pmax[i] - pmin[i] + 1);
		}
	}
}
void composite(cuComplex	*input,
			float			*output,
			const uint		frame_res,
			ushort			pmin_r,
			ushort			pmax_r,
			float			weight_r,
			ushort			pmin_g,
			ushort			pmax_g,
			float			weight_g,
			ushort			pmin_b,
			ushort			pmax_b,
			float			weight_b)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_composite << <blocks, threads, 0, 0 >> > (input,
		output,
		frame_res,
		pmin_r,
		pmax_r,
		weight_r,
		pmin_g,
		pmax_g,
		weight_g,
		pmin_b,
		pmax_b,
		weight_b);
}

/*! \brief Kernel function wrapped in float_to_ushort, making
 ** the call easier
 **/
static __global__
void kernel_float_to_ushort(const float	*input,
							void		*output,
							const uint	size,
							const float	depth)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		if (depth != 1.f)
		{
			ushort *out = reinterpret_cast<ushort *>(output);
			if (input[index] > 65535.f)
				out[index] = 65535;
			else if (input[index] < 0.f)
				out[index] = 0;
			else
				out[index] = static_cast<ushort>(input[index]);
		}
		else
		{
			uchar *out = reinterpret_cast<uchar *>(output);
			out[index] = static_cast<uchar>(input[index]);
		}
	}
}

void float_to_ushort(const float	*input,
					void			*output,
					const uint		size,
					const float		depth,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_float_to_ushort << <blocks, threads, 0, stream >> >(input, output, size, depth);
}

static __global__
void kernel_complex_to_ushort(const cuComplex	*input,
							uint				*output,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		ushort x = 0;
		ushort y = 0;
		if (input[index].x > 65535.0f)
			x = 65535;
		else if (input[index].x >= 1.0f)
			x = static_cast<ushort>(input[index].x * input[index].x);

		if (input[index].y > 65535.0f)
			y = 65535;
		else if (input[index].y >= 0.0f)
			y = static_cast<ushort>(input[index].y * input[index].x);

		auto& res = output[index];
		res ^= res;
		res = x << 16;
		res += y;
	}
}

void complex_to_ushort(const cuComplex	*input,
					uint				*output,
					const uint			size,
					cudaStream_t		stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_ushort << <blocks, threads, 0 >> >(input, output, size);
}

/*! \brief Memcpy of a complex sized frame into another buffer */
void complex_to_complex(const cuComplex	*input,
						ushort*			output,
						const uint		size,
						cudaStream_t	stream)
{
	cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice);
}

__global__
void kernel_buffer_size_conversion(char			*real_buffer,
								const char		*buffer,
								const size_t	frame_desc_width,
								const size_t	frame_desc_height,
								const size_t	real_frame_desc_width,
								const size_t	area)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < area)
	{
		uint x = index % real_frame_desc_width;
		uint y = index / real_frame_desc_width;
		if (y < frame_desc_height && x < frame_desc_width)
			real_buffer[index] = buffer[y * frame_desc_width + x];
	}
}

void buffer_size_conversion(char*					real_buffer,
							const char*				buffer,
							const FrameDescriptor	real_frame_desc,
							const FrameDescriptor	frame_desc)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem((frame_desc.height * real_frame_desc.width * static_cast<size_t>(frame_desc.depth)), threads);

	kernel_buffer_size_conversion << <blocks, threads, 0 >> >(	real_buffer,
																buffer,
																frame_desc.width * static_cast<uint>(frame_desc.depth),
																frame_desc.height * static_cast<uint>(frame_desc.depth),
																real_frame_desc.width * static_cast<uint>(frame_desc.depth),
																frame_desc.height * real_frame_desc.width * static_cast<size_t>(frame_desc.depth));
}

__global__
void kernel_accumulate_images(const float	*input,
							float			*output,
							const size_t	start,
							const size_t	max_elmt,
							const size_t	nb_elmt,
							const size_t	nb_pixel)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t	i = 0;
	int		pos = start;

	if (index < nb_pixel)
	{
		output[index] = 0;
		while (i++ < nb_elmt)
		{
			output[index] += input[index + pos * nb_pixel];
			if (--pos < 0)
				pos = max_elmt - 1;
		}
		output[index] /= nb_elmt;
	}
}

/*! \brief Kernel function wrapped in accumulate_images, making
** the call easier
**/
void accumulate_images(const float	*input,
					float			*output,
					const size_t	start,
					const size_t	max_elmt,
					const size_t	nb_elmt,
					const size_t	nb_pixel,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(nb_pixel, threads);

	kernel_accumulate_images << <blocks, threads, 0, stream >> >(input, output, start, max_elmt, nb_elmt, nb_pixel);
}

__global__
void kernel_normalize_images(float		*image,
							const float	max,
							const float	min,
							const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		if (min < 0.f)
			image[index] = (image[index] + fabs(min)) / (fabs(min) + max) * 65535.0f;
		else
			image[index] = (image[index] - min) / (max - min) * 65535.0f;
	}
}
