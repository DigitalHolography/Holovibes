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
#include "cuda_memory.cuh"

using camera::FrameDescriptor;

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
void kernel_complex_to_modulus_in_stft(float				*output,
							const cuComplex		*stft_buf,
							const ushort		pmin,
							const ushort		pmax,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		// We use a local variable so the global memory isn't read or written everytime.
		// Only written once at the end.
		float val = 0.0f;
		for (int i = pmin; i <= pmax; i++)
		{
			const cuComplex *current_p_frame = stft_buf + i * size;

			val += hypotf(current_p_frame[index].x, current_p_frame[index].y);
		}

		output[index] = val / (pmax - pmin + 1);

	}
}

void complex_to_modulus(float			*output,
						const cuComplex *stft_buf,
						const ushort	pmin,
						const ushort	pmax,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_modulus_in_stft << <blocks, threads, 0, stream >> >(output, stft_buf, pmin, pmax, size);
	// No sync needed since everything is run on stream 0
	cudaCheckError();
}

/* Kernel function wrapped in complex_to_squared_modulus. */
static __global__
void kernel_complex_to_squared_modulus(float				*output,
									const cuComplex		*stft_buf,
									const ushort		pmin,
									const ushort		pmax,
									const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		// We use a local variable so the global memory isn't read or written everytime.
		// Only written once at the end.
		float val = 0.0f;
		for (int i = pmin; i <= pmax; i++)
		{
			const cuComplex *current_p_frame = stft_buf + i * size;
			// square of the square root of the sum of the squares of x and y
			float tmp = hypotf(current_p_frame[index].x, current_p_frame[index].y);
			val += tmp * tmp;
		}
		output[index] = val / (pmax - pmin + 1);
	}
}

template <typename OTYPE, typename ITYPE, typename FUNC>
static __global__
void kernel_input_queue_to_input_buffer(OTYPE* output,
								const ITYPE* const input,
								FUNC convert,
								const uint frame_res,
								const int batch_size,
								const uint current_queue_index,
								const uint queue_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < frame_res)
	{
		uint frame_copied = 0;
		// j swip through the queue from it's start index to either the end of the queue or all of the batch
		for (int j = current_queue_index; j < queue_size && frame_copied < batch_size; ++j, ++frame_copied)
		{
			const float val = convert(input[index + j * frame_res]);

			output[index + frame_copied * frame_res].x = val;
			output[index + frame_copied * frame_res].y = 0.0f;
		}

		// Copy might reach end of the queue so we copy the missing frames
		for (int j = 0; frame_copied < batch_size; ++frame_copied, ++j)
		{
			float val = convert(input[index + j * frame_res]);

			output[index + frame_copied * frame_res].x = val;
			output[index + frame_copied * frame_res].y = 0.0f;
		}
	}
}

void input_queue_to_input_buffer(void* output,
								void* input,
								const uint frame_res,
								const int batch_size,
								const uint current_queue_index,
								const uint queue_size,
								const uint depth)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame_res, threads);

	/* Best way we found to pass function to kernels
	*  We can't declare the lambda outside this function for some reason
	* To pass lambda like that, we need to add the --extended-lambda  flag
	*/
	static const auto convert_8_bit = [] __device__ (const uchar input_pixel){ return static_cast<float>(input_pixel * 257); };
	static const auto convert_16_bit = [] __device__ (const ushort input_pixel){ return static_cast<float>(input_pixel); };
	static const auto convert_32_bit = [] __device__ (const float input_pixel){ return input_pixel; };

	switch(depth)
	{
		case 1:
			kernel_input_queue_to_input_buffer<cuComplex, uchar>
												<<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
																reinterpret_cast<uchar*>(input),
																convert_8_bit,
																frame_res,
																batch_size,
																current_queue_index,
																queue_size);
			break;
		case 2:
			kernel_input_queue_to_input_buffer<cuComplex, ushort>
												<<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
																reinterpret_cast<ushort*>(input),
																convert_16_bit,
																frame_res,
																batch_size,
																current_queue_index,
																queue_size);
			break;
		case 4:
			kernel_input_queue_to_input_buffer<cuComplex, float>
												<<<blocks, threads>>>(reinterpret_cast<cuComplex*>(output),
																reinterpret_cast<float*>(input),
																convert_32_bit,
																frame_res,
																batch_size,
																current_queue_index,
																queue_size);
			break;
	}
	// No sync needed since next call (fft1 is called on default main stream (0))
	cudaCheckError();
}

void complex_to_squared_modulus(float			*output,
								const cuComplex	*stft_buf,
								const ushort	pmin,
								const ushort	pmax,
								const uint		size,
								cudaStream_t	stream)
{
const uint threads = get_max_threads_1d();
const uint blocks = map_blocks_to_problem(size, threads);

kernel_complex_to_squared_modulus << <blocks, threads, 0, stream >> >(output, stft_buf, pmin, pmax, size);
cudaDeviceSynchronize();
cudaCheckError();
}

/* Kernel function wrapped in complex_to_argument. */
static __global__
void kernel_complex_to_argument(float			*output,
								const cuComplex	*stft_buf,
								const ushort	pmin,
								const ushort	pmax,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		// We use a local variable so the global memory isn't read or written everytime.
		// Only written once at the end.
		float val = 0.0f;
		for (int i = pmin; i <= pmax; i++)
		{
			const cuComplex *current_p_frame = stft_buf + i * size;
			// Computes the arc tangent of y / x
			// We use std::atan2 in order to obtain results in [-pi; pi].
			val += std::atan2(current_p_frame[index].y, current_p_frame[index].x);
		}
		output[index] = val / (pmax - pmin + 1);
	}
}

void complex_to_argument(float			*output,
						const cuComplex	*stft_buf,
						const ushort	pmin,
						const ushort	pmax,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_argument << <blocks, threads, 0, stream >> >(output, stft_buf, pmin, pmax, size);
	cudaDeviceSynchronize();
	cudaCheckError();
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

	// TODO : See if gpu_postprocess_frame could be used directly.
	cudaXMemcpy(output, input, sizeof(float) * size, cudaMemcpyDeviceToDevice);

	// Computing minimum and maximum values, in order to rescale properly.
	float* gpu_local_min;
	float* gpu_local_max;
	const uint float_blocks = sizeof(float) * blocks;
	if (cudaMalloc(&gpu_local_min, float_blocks) != cudaSuccess)
		return;
	if (cudaMalloc(&gpu_local_max, float_blocks) != cudaSuccess)
	{
		cudaXFree(gpu_local_min);
		return;
	}

	/* We have to hardcode the template parameter, unfortunately.
	 * It must be equal to the number of threads per block. */
	kernel_minmax <128> << <blocks, threads, threads << 1, stream >> > (output, size, gpu_local_min, gpu_local_max);
	cudaDeviceSynchronize();
	cudaCheckError();

	float	*cpu_local_min = new float[blocks];
	float	*cpu_local_max = new float[blocks];
	cudaXMemcpy(cpu_local_min, gpu_local_min, float_blocks, cudaMemcpyDeviceToHost);
	cudaXMemcpy(cpu_local_max, gpu_local_max, float_blocks, cudaMemcpyDeviceToHost);

	const float max_intensity = 65535.f;
	kernel_rescale << <blocks, threads, 0, stream >> >(	output,
														size,
														*(std::min_element(cpu_local_min, cpu_local_min + threads)),
														*(std::max_element(cpu_local_max, cpu_local_max + threads)),
														max_intensity);
	cudaDeviceSynchronize();
	cudaCheckError();
	delete[] cpu_local_max;
	delete[] cpu_local_min;
	cudaXFree(gpu_local_min);
	cudaXFree(gpu_local_max);
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
	cudaXMemcpy(cpu_buffer, input, float_frame_res, cudaMemcpyDeviceToHost);
	auto minmax = std::minmax_element(cpu_buffer, cpu_buffer + frame_res);
	min = *minmax.first;
	max = *minmax.second;

	cudaXMemcpy(output, input, float_frame_res, cudaMemcpyDeviceToDevice);

	kernel_normalize_images <<< blocks, threads, 0, stream >>> (
		output,
		max,
		min,
		frame_res);
	cudaDeviceSynchronize();
	cudaCheckError();
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
	cudaDeviceSynchronize();
	cudaCheckError();
}

/*! \brief Kernel function wrapped in endianness_conversion, making
 ** the call easier
 **/
static __global__
void kernel_endianness_conversion(const ushort	*input,
								ushort			*output,
								const uint 		batch_size,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		for (uint i = 0; i < batch_size; i++)
		{
			const uint batch_index = index + i * size;
			output[batch_index] = (input[batch_index] << 8) | (input[batch_index] >> 8);
		}
	}
}

void endianness_conversion(const ushort	*input,
						ushort			*output,
						const uint 		batch_size,
						const uint		size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_endianness_conversion << <blocks, threads, 0, stream >> >(input, output, batch_size, size);
	cudaDeviceSynchronize();
	cudaCheckError();
}

/*! \brief Kernel function wrapped in float_to_ushort, making
 ** the call easier
 **/
static __global__
void kernel_float_to_ushort(const float		*input,
							void			*output,
							const uint		size,
							const size_t	depth,
							ushort			shift = 0)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		if (depth != 1)
		{
			ushort *out = reinterpret_cast<ushort *>(output);
			if (input[index] > 65535.f)
				out[index] = 65535;
			else if (input[index] < 0.f)
				out[index] = 0;
			else
				out[index] = static_cast<ushort>(input[index]) << shift;
		}
		else
		{
			uchar *out = reinterpret_cast<uchar *>(output);
			out[index] = static_cast<uchar>(input[index]) << shift;
		}
	}
}

void float_to_ushort(const float	*input,
					void			*output,
					const uint		size,
					const size_t	depth,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_float_to_ushort << <blocks, threads, 0, stream >> >(input, output, size, depth);
	cudaDeviceSynchronize();
	cudaCheckError();
}


/*! \brief Kernel function wrapped in float_to_UINT8
**/
static __global__
void kernel_float_to_uint8(const float	*input,
	Npp8u *output,
	const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
			if (input[index] > 255.f)
				output[index] = 255;
			else if (input[index] < 0.f)
				output[index] = 0;
			else
				output[index] = static_cast<Npp8u>(input[index]);
	}
}



void float_to_uint8(const float	*input,
	Npp8u *output,
	const uint size)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_float_to_uint8 << <blocks, threads, 0, 0 >> >(input, output, size);
	cudaDeviceSynchronize();
	cudaCheckError();
}

/*! \brief Kernel function wrapped in float_to_UINT8
**/
static __global__
void kernel_uint8_to_float(const  Npp8u	*input,
	float *output,
	const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
		output[index] = static_cast<float>(input[index]);
}

void uint8_to_float(const  Npp8u	*input,
	float			*output,
	const uint		size)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_uint8_to_float << <blocks, threads, 0, 0 >> >(input, output, size);
	cudaDeviceSynchronize();
	cudaCheckError();
}

static __global__
void kernel_ushort_to_uchar(const ushort	*input,
	uchar		*output,
	const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
		output[index] = input[index] >> 8;
}

void ushort_to_uchar(const ushort	*input,
	uchar			*output,
	const uint		size,
	cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_ushort_to_uchar << <blocks, threads, 0, stream >> >(input, output, size);
	cudaDeviceSynchronize();
	cudaCheckError();
}

static __global__
void kernel_complex_to_ushort(const cuComplex	*input,
							uint				*output,
							const uint			size,
							ushort				shift = 0)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		cuComplex pixel = input[index];
		ushort x = 0;
		ushort y = 0;
		if (pixel.x > 65535.0f)
			x = 65535;
		else if (pixel.x >= 1.0f)
			x = static_cast<ushort>(pixel.x);

		if (pixel.y > 65535.0f)
			y = 65535;
		else if (pixel.y >= 0.0f)
			y = static_cast<ushort>(pixel.y);

		auto& res = output[index];
		res = x << 16;
		res |= y;
		res <<= shift;
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
	cudaDeviceSynchronize();
	cudaCheckError();
}

/*! \brief Memcpy of a complex sized frame into another buffer */
void complex_to_complex(const cuComplex	*input,
						ushort*			output,
						const uint		size,
						cudaStream_t	stream)
{
	cudaXMemcpy(output, input, size, cudaMemcpyDeviceToDevice);
}

__global__
void kernel_buffer_size_conversion(char			*real_buffer,
								const char		*buffer,
								const size_t	fd_width,
								const size_t	fd_height,
								const size_t	real_fd_width,
								const size_t	area)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < area)
	{
		uint x = index % real_fd_width;
		uint y = index / real_fd_width;
		if (y < fd_height && x < fd_width)
			real_buffer[index] = buffer[y * fd_width + x];
	}
}

void buffer_size_conversion(char*					real_buffer,
							const char*				buffer,
							const FrameDescriptor	real_fd,
							const FrameDescriptor	fd)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem((fd.height * real_fd.width * static_cast<size_t>(fd.depth)), threads);

	kernel_buffer_size_conversion << <blocks, threads, 0 >> >(	real_buffer,
																buffer,
																fd.width * fd.depth,
																fd.height * fd.depth,
																real_fd.width * fd.depth,
																fd.height * real_fd.width * static_cast<size_t>(fd.depth));
	cudaCheckError();
}

__global__
void kernel_accumulate_images(const float	*input,
							float			*output,
							const size_t	end,
							const size_t	max_elmt,
							const size_t	nb_elmt,
							const size_t	nb_pixel)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	long int pos = end; // end is excluded

	if (index < nb_pixel)
	{
		float val = 0;
		for (size_t i = 0; i < nb_elmt; i++)
		{
			// get last index when pos is out of range
			// reminder: the given input is from ciruclar queue
			pos--;
			if (pos < 0)
				pos = max_elmt - 1;

			val += input[index + pos * nb_pixel];
		}
		output[index] = val / nb_elmt;
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
	cudaDeviceSynchronize();
	cudaCheckError();
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

__global__
void kernel_normalize_complex(cuComplex		*image,
							const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		image[index].x += 1;
		image[index].x *= 65535 / 2;
		image[index].y += 1;
		image[index].y *= 65535 / 2;
	}
}

void normalize_complex(cuComplex *image,
 const uint	size)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(size, threads);
	kernel_normalize_complex << <blocks, threads, 0, 0 >> > (image, size);
	cudaDeviceSynchronize();
	cudaCheckError();
}

template<typename T>
__global__
void kernel_convert_frame_for_display(const T     	*input,
	void     		*output,
	const uint	    size,
	const uint     depth,
	const ushort	shift)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		T* out = reinterpret_cast<T*>(output);
		out[index] = input[index] << shift;
	}
}

void convert_frame_for_display(const void   	*input,
	void			*output,
	const uint	size,
	const uint    depth,
	const ushort	shift)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	if (depth == 8)
	{
		kernel_complex_to_ushort << <blocks, threads, 0 >> > (static_cast<const cuComplex*>(input), static_cast<uint*>(output), size, shift);
	}
	else if (depth == 4)
	{
		kernel_float_to_ushort << <blocks, threads, 0, 0 >> > (static_cast<const float*>(input), output, size, depth, shift);
	}
	else if (depth == 2)
	{
		kernel_convert_frame_for_display<ushort> << <blocks, threads, 0, 0 >> > (static_cast<const ushort*>(input), output, size, depth, shift);
	}
	else if (depth == 1)
	{
		kernel_convert_frame_for_display<uchar> << <blocks, threads, 0, 0 >> > (static_cast<const uchar*>(input), output, size, depth, shift);
	}

	cudaDeviceSynchronize();
	cudaCheckError();
}

void frame_to_complex(void *input, cufftComplex *output, const uint frame_res, const uint depth)
{
	const uint				threads = get_max_threads_1d();
	const uint				blocks = map_blocks_to_problem(frame_res, threads);

	switch (depth)
	{
	case 1:
		img8_to_complex <<<blocks, threads>>> (
			output,
			static_cast<uchar*>(input),
			frame_res);
		break;
	case 2:
		img16_to_complex <<<blocks, threads>>> (
			output,
			static_cast<ushort*>(input),
			frame_res);
		break;
	case 4:
		float_to_complex <<<blocks, threads>>> (
			output,
			static_cast<float*>(input),
			frame_res);
		break;
	case 8:
		cudaXMemcpy(output, input, frame_res << 3, cudaMemcpyDeviceToDevice);
		break;
	default:
		break;
	}

	cudaDeviceSynchronize();
	cudaCheckError();
}