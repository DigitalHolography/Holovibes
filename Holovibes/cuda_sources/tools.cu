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

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_unwrap.cuh"

__global__
void kernel_apply_lens(cuComplex		*input,
					const uint			input_size,
					const cuComplex		*lens,
					const uint			lens_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < input_size)
	{
		uint	index2 = index % lens_size; // necessary if more than one frame
		float	tmp_x = input[index].x;
		input[index].x = input[index].x * lens[index2].x - input[index].y * lens[index2].y;
		input[index].y = input[index].y * lens[index2].x + tmp_x * lens[index2].y;
		//index += blockDim.x * gridDim.x;
	}
}

static __global__
void kernel_shift_corners(float		*input,
						const uint	size_x,
						const uint	size_y)
{
	const uint	i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint	j = blockIdx.y * blockDim.y + threadIdx.y;
	const uint	index = j * blockDim.x * gridDim.x + i;
	uint	ni = 0;
	uint	nj = 0;
	uint	nindex = 0;

	// Superior half of the matrix
	const uint size_x2 = size_x >> 1;
	const uint size_y2 = size_y >> 1;
	if (j >= size_y2)
	{
		// Left superior quarter of the matrix
		if (i < size_x2)
			ni = i + size_x2;
		else // Right superior quarter
			ni = i - size_x2;
		nj = j - size_y2;
		nindex = nj * size_x + ni;

		float tmp = input[nindex];
		input[nindex] = input[index];
		input[index] = tmp;
	}
}

void shift_corners(float		*input,
				const uint		size_x,
				const uint		size_y,
				cudaStream_t	stream)
{
	uint threads_2d = get_max_threads_2d();
	dim3 lthreads(threads_2d, threads_2d);
	dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

	kernel_shift_corners << < lblocks, lthreads, 0, stream >> >(input, size_x, size_y);
}

/* Kernel used in apply_log10 */
static __global__
void kernel_log10(float		*input,
				const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		input[index] = log10f(input[index]);
	//	index += blockDim.x * gridDim.x;
	}
}

void apply_log10(float			*input,
				const uint		size,
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	kernel_log10 << <blocks, threads, 0, stream >> >(input, size);
}

/* Kernel used in convolution_operator */
static __global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < size)
	{
		output[index] = hypotf(input[index].x, input[index].y);
		//index += blockDim.x * gridDim.x;
	}
}

void demodulation(cuComplex			*input,
				const cufftHandle	plan1d,
				cudaStream_t		stream)
{
	// FFT 1D TEMPORAL
	cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);
}


void convolution_operator(	const cuComplex		*x,
							const cuComplex		*k,
							float				*out,
							const uint			size,
							const cufftHandle	plan2d_x,
							const cufftHandle	plan2d_k,
							cudaStream_t		stream)
{
	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	/* The convolution operator is used only when using autofocus feature.
	 * It could be optimized but it's useless since it will be used occasionnally. */
	cuComplex *tmp_x;
	cuComplex *tmp_k;
	uint	complex_size = size * sizeof(cuComplex);
	if (cudaMalloc<complex>(&tmp_x, complex_size) != cudaSuccess)
		return;
	if (cudaMalloc<complex>(&tmp_k, complex_size) != cudaSuccess)
		return;

	cufftExecC2C(plan2d_x, const_cast<cuComplex*>(x), tmp_x, CUFFT_FORWARD);
	cufftExecC2C(plan2d_k, const_cast<cuComplex*>(k), tmp_k, CUFFT_FORWARD);

	cudaStreamSynchronize(stream);

	kernel_multiply_frames_complex << <blocks, threads, 0, stream >> >(tmp_x, tmp_k, tmp_x, size);

	cudaStreamSynchronize(stream);

	cufftExecC2C(plan2d_x, tmp_x, tmp_x, CUFFT_INVERSE);

	cudaStreamSynchronize(stream);

	kernel_complex_to_modulus << <blocks, threads, 0, stream >> >(tmp_x, out, size);

	cudaFree(tmp_x);
	cudaFree(tmp_k);
}

void frame_memcpy(float				*input,
				const Rectangle&	zone,
				const uint			input_width,
				float				*output,
				const uint			output_width,
				cudaStream_t		stream)
{
	const float	*zone_ptr = input + (zone.topLeft().y() * input_width + zone.topLeft().x());
	const uint	output_width_float = output_width * sizeof(float);
	cudaMemcpy2DAsync(	output,
						output_width_float,
						zone_ptr,
						input_width * sizeof(float),
						output_width_float,
						output_width,
						cudaMemcpyDeviceToDevice,
						stream);
	cudaStreamSynchronize(stream);
}

/* Kernel helper used in average.
 *
 * Sums up the *size* first elements of input and stores the result in sum.
 *
 * SpanSize is the number of elements processed by a single thread.
 * This way of doing things comes from the empirical fact that (at the point
 * of this writing) loop unrolling in CUDA kernels may prove more efficient,
 * when the operation is really small. */
template <uint SpanSize>

static __global__
void kernel_sum(const float* input, float* sum, const size_t size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if ((index + SpanSize - 1) < size && (index % SpanSize) == 0)
	{
		float tmp_reduce = 0.0f;
		for (uint i = 0; i < SpanSize; ++i)
			tmp_reduce += input[index + i];
		// Atomic operation is needed here to guarantee a correct value.
		atomicAdd(sum, tmp_reduce);
	}
}

float average_operator(const float	*input,
					const uint		size,
					cudaStream_t	stream)
{
	const uint	threads = THREADS_128;
	uint		blocks = map_blocks_to_problem(size, threads);
	float		*gpu_sum;
	float		cpu_sum = 0.0f;

	if (cudaMalloc<float>(&gpu_sum, sizeof(float)) == cudaSuccess)
		cudaMemsetAsync(gpu_sum, 0, sizeof(float), stream);
	else
		return (0.f);
	cudaStreamSynchronize(stream);

	// A SpanSize of 4 has been determined to be an optimal choice here.
	kernel_sum <4> << <blocks, threads, 0, stream >> >(
		input,
		gpu_sum,
		size);
	cudaMemcpyAsync(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(stream);

	cudaFree(gpu_sum);

	return cpu_sum /= static_cast<float>(size);
}

void phase_increase(const cuComplex			*cur,
					UnwrappingResources		*resources,
					const size_t			image_size)
{
	const uint	threads = THREADS_128; // 3072 cuda cores / 24 SMM = 128 Threads per SMM
	const uint	blocks = map_blocks_to_problem(image_size, threads);
	static bool first_time = true;
	if (first_time)
	{
		cudaMemcpy(	resources->gpu_predecessor_,
					cur,
					sizeof(cuComplex)* image_size,
					cudaMemcpyDeviceToDevice);
		first_time = false;
	}

	// Compute the newest phase image, not unwrapped yet
	kernel_compute_angle_mult << <blocks, threads >> >(	resources->gpu_predecessor_,
														cur,
														resources->gpu_angle_current_,
														image_size);
	// Updating predecessor (complex image) for the next iteration
	cudaMemcpy(	resources->gpu_predecessor_,
				cur,
				sizeof(cuComplex) * image_size,
				cudaMemcpyDeviceToDevice);

	/* Copying in order to later enqueue the (not summed up with values
	 * in gpu_unwrap_buffer_) phase image. */
	cudaMemcpy(	resources->gpu_angle_copy_,
				resources->gpu_angle_current_,
				sizeof(float) * image_size,
				cudaMemcpyDeviceToDevice);

	// Applying history on the latest phase image
	kernel_correct_angles << <blocks, threads >> >(	resources->gpu_angle_current_,
													resources->gpu_unwrap_buffer_,
													image_size,
													resources->size_);

	/* Store the new phase image in the next buffer position.
	* The buffer is handled as a circular buffer. */
	float	*next_unwrap = resources->gpu_unwrap_buffer_ + image_size * resources->next_index_;
	cudaMemcpy(	next_unwrap,
				resources->gpu_angle_copy_,
				sizeof(float)* image_size,
				cudaMemcpyDeviceToDevice);
	if (resources->size_ < resources->capacity_)
		++resources->size_;
	resources->next_index_ = (resources->next_index_ + 1) % resources->capacity_;
}

void unwrap_2d(	float*						input,
				const cufftHandle			plan2d,
				UnwrappingResources_2d*		res,
				FrameDescriptor&	fd,
				float*						output,
				cudaStream_t				stream)
{
	uint		threads_2d = get_max_threads_2d();
	dim3		lthreads(threads_2d, threads_2d);
	dim3		lblocks(fd.width / threads_2d, fd.height / threads_2d);
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(res->image_resolution_, threads);

	kernel_init_unwrap_2d << < lblocks, lthreads, 0, stream >> > (	fd.width,
																	fd.height,
																	fd.frame_res(),
																	input,
																	res->gpu_fx_,
																	res->gpu_fy_,
																	res->gpu_z_);
	ushort middlex = fd.width >> 1;
	ushort middley = fd.height >> 1;
	circ_shift_float << < blocks, threads, 0, stream >> > (	res->gpu_fx_,
															res->gpu_shift_fx_,
															middlex,
															middley,
															fd.width,
															fd.height,
															fd.frame_res());
	circ_shift_float << < blocks, threads, 0, stream >> > (	res->gpu_fy_,
															res->gpu_shift_fy_,
															middlex,
															middley,
															fd.width,
															fd.height,
															fd.frame_res());
	gradient_unwrap_2d(plan2d, res, fd, stream);
	eq_unwrap_2d(plan2d, res, fd, stream);
	phi_unwrap_2d(plan2d, res, fd, output, stream);
}

void gradient_unwrap_2d(const cufftHandle			plan2d,
						UnwrappingResources_2d*		res,
						FrameDescriptor&			fd,
						cudaStream_t				stream)
{
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(res->image_resolution_, threads);
	cuComplex	single_complex = make_cuComplex(0.f, static_cast<float>(M_2PI));

	cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
	cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
	kernel_multiply_complexes_by_floats_ << < blocks, threads, 0, stream >> >(	res->gpu_shift_fx_,
																				res->gpu_shift_fy_,
																				res->gpu_grad_eq_x_,
																				res->gpu_grad_eq_y_,
																				fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_INVERSE);
	kernel_multiply_complexes_by_single_complex << < blocks, threads, 0, stream >> >(	res->gpu_grad_eq_x_,
																						res->gpu_grad_eq_y_,
																						single_complex,
																						fd.frame_res());
}

void eq_unwrap_2d(const cufftHandle			plan2d,
				UnwrappingResources_2d*		res,
				FrameDescriptor&			fd,
				cudaStream_t				stream)
{
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(res->image_resolution_, threads);
	cuComplex	single_complex = make_cuComplex(0, 1);

	kernel_multiply_complex_by_single_complex << < blocks, threads, 0, stream >> >(	res->gpu_z_,
																					single_complex,
																					fd.frame_res());
	kernel_conjugate_complex << < blocks, threads, 0, stream >> >(res->gpu_z_, fd.frame_res());
	kernel_multiply_complex_frames_by_complex_frame << < blocks, threads, 0, stream >> >(	res->gpu_grad_eq_x_,
																							res->gpu_grad_eq_y_,
																							res->gpu_z_,
																							fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
	kernel_norm_ratio << < blocks, threads, 0, stream >> >(	res->gpu_shift_fx_,
															res->gpu_shift_fy_,
															res->gpu_grad_eq_x_,
															res->gpu_grad_eq_y_,
															fd.frame_res());
}

void phi_unwrap_2d(	const cufftHandle			plan2d,
					UnwrappingResources_2d*		res,
					FrameDescriptor&			fd,
					float*						output,
					cudaStream_t				stream)
{
	const uint threads = THREADS_128;
	const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);

	//	kernel_convergence << < 1, 1, 0, stream >> >(res->gpu_grad_eq_x_,
	//		res->gpu_grad_eq_y_);
	kernel_add_complex_frames << < blocks, threads, 0, stream >> >(res->gpu_grad_eq_x_, res->gpu_grad_eq_y_, fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	kernel_unwrap2d_last_step << < blocks, threads, 0, stream >> > (output, res->gpu_grad_eq_x_, fd.frame_res());
}

__global__
void circ_shift(cuComplex	*input,
				cuComplex	*output,
				const int	i, // shift on x axis
				const int	j, // shift on y axis
				const uint	width,
				const uint	height,
				const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	int		index_x = 0;
	int		index_y = 0;
	int		shift_x = 0;
	int		shift_y = 0;
	// In ROI
	//while (index < size)
	{
		index_x = index % width;
		index_y = index / height;
		shift_x = index_x - i;
		shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;
		output[(width * shift_y) + shift_x] = input[index];
		//index += blockDim.x * gridDim.x;
	}
}

__global__
void circ_shift_float(float		*input,
					float		*output,
					const int	i, // shift on x axis
					const int	j, // shift on y axis
					const uint	width,
					const uint	height,
					const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	int		index_x = 0;
	int		index_y = 0;
	int		shift_x = 0;
	int		shift_y = 0;
	// In ROI
	//while (index < size)
	{
		index_x = index % width;
		index_y = index / height;
		shift_x = index_x - i;
		shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;
		output[(width * shift_y) + shift_x] = input[index];
		//index += blockDim.x * gridDim.x;
	}
}
