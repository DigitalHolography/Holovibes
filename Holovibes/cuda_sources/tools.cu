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
#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/array.hh"
#include "cuda_tools/cufft_handle.hh"
#include "logger.hh"

using camera::FrameDescriptor;
using namespace holovibes;
using cuda_tools::UniquePtr;
using cuda_tools::Array;
using cuda_tools::CufftHandle;

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
	cudaCheckError();
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
	cudaCheckError();
}

/* Kernel used in convolution_operator */
__global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
		output[index] = hypotf(input[index].x, input[index].y);
}

void demodulation(cuComplex			*input,
				const cufftHandle	plan1d,
				cudaStream_t		stream)
{
	// FFT 1D TEMPORAL
	cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);
}


void convolution_float(		const float			*a,
							const float			*b,
							float				*out,
							const uint			size,
							const cufftHandle	plan2d_a,
							const cufftHandle	plan2d_b,
							const cufftHandle	plan2d_inverse,
							cudaStream_t		stream)
{
	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	// The convolution operator could be optimized.
	// TODO: pre allocate tmp buffers and pass them to the function
	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_a(size);
	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_b(size);
	if (!tmp_a || !tmp_b)
		return;
	
	cufftExecR2C(plan2d_a, const_cast<float*>(a), tmp_a.get());
	cufftExecR2C(plan2d_b, const_cast<float*>(b), tmp_b.get());
	

	cudaStreamSynchronize(0);
	kernel_multiply_frames_complex <<<blocks, threads, 0, stream >>>(tmp_a.get(), tmp_b.get(), tmp_a.get(), size);
	cudaCheckError();

	cudaStreamSynchronize(stream);

	cufftExecC2R(plan2d_inverse, tmp_a.get(), out);

	cudaStreamSynchronize(0);

	//kernel_complex_to_modulus <<<blocks, threads, 0, stream >>>(tmp_a, out, size);
	//cudaStreamSynchronize(stream);
}


void convolution_operator(	const cuComplex		*a,
							const cuComplex		*b,
							float				*out,
							const uint			size,
							const cufftHandle	plan2d_a,
							const cufftHandle	plan2d_b,
							cudaStream_t		stream)
{
	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	/* The convolution operator could be optimized. */

	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_a(size);
	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_b(size);
	if (!tmp_a || !tmp_b)
		return;
	
	cufftExecC2C(plan2d_a, const_cast<cuComplex*>(a), tmp_a.get(), CUFFT_FORWARD);
	cufftExecC2C(plan2d_b, const_cast<cuComplex*>(b), tmp_b.get(), CUFFT_FORWARD);
	
    /*float* abs_a = (float*)malloc(size * sizeof(float));
	float* abs_b = (float*)malloc(size * sizeof(float));
	kernel_complex_to_modulus <<<blocks, threads, 0, stream >>>(tmp_a, abs_a, size);
	kernel_complex_to_modulus <<<blocks, threads, 0, stream >>>(tmp_b, abs_b, size);
	cufftExecR2C(plan2d_a, abs_a, tmp_a);
	cufftExecR2C(plan2d_b, abs_b, tmp_b);

	free(abs_a);
	free(abs_b);*/

	cudaStreamSynchronize(stream);
	kernel_multiply_frames_complex <<<blocks, threads, 0, stream >>>(tmp_a.get(), tmp_b.get(), tmp_a.get(), size);
	cudaCheckError();

	cudaStreamSynchronize(stream);

	cufftExecC2C(plan2d_a, tmp_a.get(), tmp_a.get(), CUFFT_INVERSE);

	cudaStreamSynchronize(stream);

	kernel_complex_to_modulus <<<blocks, threads, 0, stream >>>(tmp_a.get(), out, size);
	cudaCheckError();

	cudaStreamSynchronize(stream);
}

void frame_memcpy(float				*input,
				const units::RectFd&	zone,
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

/*
* Kernel helper used in average.
 *
 * Sums up the *size* real parts of first elements of input and stores the result in sum.
 *
 * SpanSize is the number of elements processed by a single thread.
 * This way of doing things comes from the empirical fact that (at the point
 * of this writing) loop unrolling in CUDA kernels may prove more efficient,
 * when the operation is really small.
*/
template <uint SpanSize>

static __global__
void kernel_sum_of_real_parts(const cufftComplex* input, float* sum, const size_t size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if ((index + SpanSize - 1) < size && (index % SpanSize) == 0)
	{
		float tmp_reduce = 0.0f;
		for (uint i = 0; i < SpanSize; ++i)
			tmp_reduce += input[index + i].x;
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
		return 0.f;
	cudaStreamSynchronize(stream);

	// A SpanSize of 4 has been determined to be an optimal choice here.
	kernel_sum <4> << <blocks, threads, 0, stream >> >(
		input,
		gpu_sum,
		size);
	cudaCheckError();
	cudaMemcpyAsync(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaStreamSynchronize(stream);

	cudaFree(gpu_sum);

	return cpu_sum /= static_cast<float>(size);
}


float average_operator_from_complex(const cufftComplex *input,
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
		return 0.f;
	cudaStreamSynchronize(stream);

	// A SpanSize of 4 has been determined to be an optimal choice here.
	kernel_sum_of_real_parts <4> << <blocks, threads, 0, stream >> >(
		input,
		gpu_sum,
		size);
	cudaCheckError();
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
	cudaCheckError();
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
	cudaCheckError();

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
				const FrameDescriptor&		fd,
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
	cudaCheckError();
	ushort middlex = fd.width >> 1;
	ushort middley = fd.height >> 1;
	circ_shift_float << < blocks, threads, 0, stream >> > (	res->gpu_fx_,
															res->gpu_shift_fx_,
															middlex,
															middley,
															fd.width,
															fd.height,
															fd.frame_res());
	cudaCheckError();
	circ_shift_float << < blocks, threads, 0, stream >> > (	res->gpu_fy_,
															res->gpu_shift_fy_,
															middlex,
															middley,
															fd.width,
															fd.height,
															fd.frame_res());
	cudaCheckError();
	gradient_unwrap_2d(plan2d, res, fd, stream);
	eq_unwrap_2d(plan2d, res, fd, stream);
	phi_unwrap_2d(plan2d, res, fd, output, stream);
}

void gradient_unwrap_2d(const cufftHandle			plan2d,
						UnwrappingResources_2d*		res,
						const FrameDescriptor&			fd,
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
	cudaCheckError();
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_INVERSE);
	kernel_multiply_complexes_by_single_complex << < blocks, threads, 0, stream >> >(	res->gpu_grad_eq_x_,
																						res->gpu_grad_eq_y_,
																						single_complex,
																						fd.frame_res());
	cudaCheckError();
}

void eq_unwrap_2d(const cufftHandle			plan2d,
				UnwrappingResources_2d*		res,
				const FrameDescriptor&			fd,
				cudaStream_t				stream)
{
	const uint	threads = THREADS_128;
	const uint	blocks = map_blocks_to_problem(res->image_resolution_, threads);
	cuComplex	single_complex = make_cuComplex(0, 1);

	kernel_multiply_complex_by_single_complex << < blocks, threads, 0, stream >> >(	res->gpu_z_,
																					single_complex,
																					fd.frame_res());
	cudaCheckError();
	kernel_conjugate_complex << < blocks, threads, 0, stream >> >(res->gpu_z_, fd.frame_res());
	cudaCheckError();
	kernel_multiply_complex_frames_by_complex_frame << < blocks, threads, 0, stream >> >(	res->gpu_grad_eq_x_,
																							res->gpu_grad_eq_y_,
																							res->gpu_z_,
																							fd.frame_res());
	cudaCheckError();
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
	kernel_norm_ratio << < blocks, threads, 0, stream >> >(	res->gpu_shift_fx_,
															res->gpu_shift_fy_,
															res->gpu_grad_eq_x_,
															res->gpu_grad_eq_y_,
															fd.frame_res());
	cudaCheckError();
}

void phi_unwrap_2d(	const cufftHandle			plan2d,
					UnwrappingResources_2d*		res,
					const FrameDescriptor&			fd,
					float*						output,
					cudaStream_t				stream)
{
	const uint threads = THREADS_128;
	const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);

	//	kernel_convergence << < 1, 1, 0, stream >> >(res->gpu_grad_eq_x_,
	//		res->gpu_grad_eq_y_);
	kernel_add_complex_frames << < blocks, threads, 0, stream >> >(res->gpu_grad_eq_x_, res->gpu_grad_eq_y_, fd.frame_res());
	cudaCheckError();
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	kernel_unwrap2d_last_step << < blocks, threads, 0, stream >> > (output, res->gpu_grad_eq_x_, fd.frame_res());
	cudaCheckError();
}

__global__
void circ_shift(const cuComplex	*input,
				cuComplex	*output,
				const int	i, // shift on x axis
				const int	j, // shift on y axis
				const uint	width,
				const uint	height,
				const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		int index_x = index % width;
		int index_y = index / width;
		int shift_x = index_x - i;
		int shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;
        auto rhs = input[index];
		output[(width * shift_y) + shift_x] = rhs;
	}
}

__global__
void circ_shift_float(const float		*input,
					float		*output,
					const int	i, // shift on x axis
					const int	j, // shift on y axis
					const uint	width,
					const uint	height,
					const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		int index_x = index % width;
		int index_y = index / width;
		int shift_x = index_x - i;
		int shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;
        auto rhs = input[index];
		output[(width * shift_y) + shift_x] = rhs;
	}
}

__global__
void kernel_translation(float		*input,
						float		*output,
						uint		width,
						uint		height,
						int			shift_x,
						int			shift_y)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < width * height)
	{
		const int new_x = index % width;
		const int new_y = index / width;
		const int old_x = (new_x - shift_x + width) % width;
		const int old_y = (new_y - shift_y + height) % height;
		output[index] = input[old_y * width + old_x];
	}
}

// TODO: change name (array_circshift)
void complex_translation(float		*frame,
						uint		width,
						uint		height,
						int			shift_x,
						int			shift_y)
{
	// We have to use a temporary buffer to avoid overwriting pixels that haven't moved yet
	float *tmp_buffer;
	if (cudaMalloc(&tmp_buffer, width * height * sizeof(float)) != cudaSuccess)
	{
		LOG_ERROR("Can't callocate buffer for repositioning");
		return;
	}

	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(width * height, threads);

	kernel_translation<<<blocks, threads, 0, 0>>>(frame, tmp_buffer, width, height, shift_x, shift_y);
	cudaCheckError();
	cudaStreamSynchronize(0);
	cudaMemcpy(frame, tmp_buffer, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(tmp_buffer);
}

void correlation_operator(float* a, float* b, float* out, QPoint dimensions)
{
	uint size = dimensions.x() * dimensions.y();
	Array<cuComplex> tmp_a(size);
	Array<cuComplex> tmp_b(size);
	CufftHandle plan2d;
	const int width = dimensions.x();
	const int height = dimensions.y();


	plan2d.plan(width, height, CUFFT_R2C);
	cufftExecR2C(plan2d, a, tmp_a);
	cufftExecR2C(plan2d, b, tmp_b);
	cudaStreamSynchronize(0);
	cudaCheckError();

	multiply_frames_complex(tmp_a, tmp_b, tmp_a, size);
	cudaStreamSynchronize(0);

	plan2d.plan(width, height, CUFFT_C2R);

	Array<cuComplex> complex_buffer(size);

	cufftExecC2R(plan2d, tmp_a, out);
	cudaStreamSynchronize(0);
	cudaCheckError();
}

float get_norm(const float	*matrix,
			   size_t		size)
{
	uint threads = 1024;
	uint blocks = map_blocks_to_problem(size, threads);

	holovibes::cuda_tools::UniquePtr<float> output(blocks);
	normalize_float_matrix<1024><<<blocks, threads, threads * sizeof(float)>>>(matrix, output, static_cast<uint>(size));

	float *intermediate_sum_cpu = new float[blocks]; //never more than 4096 threads
	//need to be on cpu for the sum
	cudaMemcpy(intermediate_sum_cpu, output, blocks * sizeof(float), cudaMemcpyDeviceToHost); //TODO faire la somme sur GPU
	float sum = 0;
	for (uint i = 0; i < blocks; i++)
		sum += intermediate_sum_cpu[i];
	delete[] intermediate_sum_cpu;

	return sum;
}
