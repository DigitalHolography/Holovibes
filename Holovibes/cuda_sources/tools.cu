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
#include "cuda_tools/cufft_handle.hh"
#include "logger.hh"
#include "cuda_memory.cuh"
#include "Common.cuh"

#include <cassert>

using camera::FrameDescriptor;
using namespace holovibes;
using cuda_tools::UniquePtr;
using cuda_tools::CufftHandle;

__global__
void kernel_apply_lens(cuComplex		*input,
					cuComplex 			*output,
					const uint 			batch_size,
					const uint			input_size,
					const cuComplex		*lens,
					const uint			lens_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < input_size)
	{
		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * input_size;

			const float	tmp_x = input[batch_index].x;
			output[batch_index].x = input[batch_index].x * lens[index].x - input[batch_index].y * lens[index].y;
			output[batch_index].y = input[batch_index].y * lens[index].x + tmp_x * lens[index].y;
		}
	}
}

namespace
{
	template<typename T>
	__global__
	void kernel_shift_corners(const T *input,
							  T *output,
							  const uint batch_size,
							  const uint	size_x,
							  const uint	size_y)
	{
		const uint	i = blockIdx.x * blockDim.x + threadIdx.x;
		const uint	j = blockIdx.y * blockDim.y + threadIdx.y;
		const uint	index = j * (size_x) + i;
		uint	ni = 0;
		uint	nj = 0;
		uint	nindex = 0;

		const uint size_x2 = size_x / 2;
		const uint size_y2 = size_y / 2;

		// Superior half of the matrix
		if (j < size_y2)
		{
			// Left superior quarter of the matrix
			if (i < size_x2)
				ni = i + size_x2;
			else // Right superior quarter
				ni = i - size_x2;
			nj = j + size_y2;
			nindex = nj * size_x + ni;

			for (uint i = 0; i < batch_size; ++i)
			{
				const uint batch_index = index + i * size_x * size_y;
				const uint batch_nindex = nindex + i * size_x * size_y;

				// Allows output = input
				T tmp = input[batch_nindex];
				output[batch_nindex] = input[batch_index];
				output[batch_index] = tmp;
			}
		}
	}

	template<typename T>
	void shift_corners_caller(const T *input,
							  T *output,
							  const uint batch_size,
							  const uint size_x,
							  const uint size_y,
							  cudaStream_t stream)
	{
		uint threads_2d = get_max_threads_2d();
		dim3 lthreads(threads_2d, threads_2d);
		dim3 lblocks(1 + (size_x - 1) / threads_2d, 1 + (size_y - 1) / threads_2d);

		kernel_shift_corners<T> <<< lblocks, lthreads, 0, stream >> >(input, output, batch_size, size_x, size_y);
		cudaCheckError();
	}

	template<typename T>
	void shift_corners_caller(T*		input,
							  const uint batch_size,
							  const uint		size_x,
							  const uint		size_y,
							  cudaStream_t	stream)
	{
		uint threads_2d = get_max_threads_2d();
		dim3 lthreads(threads_2d, threads_2d);
		dim3 lblocks(1 + (size_x - 1) / threads_2d, 1 + (size_y - 1) / threads_2d);

		kernel_shift_corners<T> <<< lblocks, lthreads, 0, stream >> >(input, input, batch_size, size_x, size_y);
		cudaCheckError();
	}
}

void shift_corners(float3 *input,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<float3>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(const float3 *input,
				   float3 *output,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<float3>(input, output, batch_size, size_x, size_y, stream);
}

void shift_corners(float *input,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<float>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(const float *input,
				   float *output,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<float>(input, output, batch_size, size_x, size_y, stream);
}

void shift_corners(cuComplex *input,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<cuComplex>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(const cuComplex *input,
				   cuComplex *output,
				   const uint batch_size,
				   const uint size_x,
				   const uint size_y,
				   cudaStream_t stream)
{
	shift_corners_caller<cuComplex>(input, output, batch_size, size_x, size_y, stream);
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

	cufftSafeCall(cufftExecR2C(plan2d_a, const_cast<float*>(a), tmp_a.get()));
	cufftSafeCall(cufftExecR2C(plan2d_b, const_cast<float*>(b), tmp_b.get()));

	cudaStreamSynchronize(0);

	cudaXMemset(tmp_a.get(), 0, sizeof(cuComplex));
	cudaXMemset(tmp_b.get(), 0, sizeof(cuComplex));

	kernel_conjugate_complex <<<blocks, threads, 0, stream>>> (tmp_b.get(), size);

	cudaStreamSynchronize(0);
	cudaCheckError();
	kernel_multiply_frames_complex <<<blocks, threads, 0, stream >>>(tmp_a.get(), tmp_b.get(), tmp_a.get(), size);

	cudaStreamSynchronize(stream);
	cudaCheckError();

	cufftSafeCall(cufftExecC2R(plan2d_inverse, tmp_a.get(), out));

	cudaStreamSynchronize(0);
	cudaCheckError();
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

	cufftSafeCall(cufftExecC2C(plan2d_a, const_cast<cuComplex*>(a), tmp_a.get(), CUFFT_FORWARD));
	cufftSafeCall(cufftExecC2C(plan2d_b, const_cast<cuComplex*>(b), tmp_b.get(), CUFFT_FORWARD));

	cudaStreamSynchronize(0);
	kernel_multiply_frames_complex <<<blocks, threads, 0, stream >>>(tmp_a.get(), tmp_b.get(), tmp_a.get(), size);
	cudaCheckError();

	cudaStreamSynchronize(stream);

	cufftSafeCall(cufftExecC2C(plan2d_a, tmp_a.get(), tmp_a.get(), CUFFT_INVERSE));

	cudaStreamSynchronize(0);

	kernel_complex_to_modulus <<<blocks, threads, 0, stream >>>(tmp_a.get(), out, size);
	cudaCheckError();

	cudaStreamSynchronize(stream);
}

void frame_memcpy(const float				*input,
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

cudaError_t embedded_frame_cpy(const char *input,
	                           const uint input_width,
	                           const uint input_height,
	                           char *output,
	                           const uint output_width,
	                           const uint output_height,
	                           const uint output_startx,
	                           const uint output_starty,
	                           const uint elm_size,
	                           cudaMemcpyKind kind,
	                           cudaStream_t stream)
{
    assert(input_width + output_startx <= output_width);
    assert(input_height + output_starty <= output_height);

    char *output_write_start = output + elm_size * (output_starty * output_width + output_startx);
    return cudaMemcpy2DAsync(output_write_start,
                             output_width * elm_size,
                             input,
                             input_width * elm_size,
                             input_width * elm_size,
                             input_height,
                             kind,
                             stream);
}

cudaError_t embed_into_square(const char *input,
							  const uint input_width,
	  						  const uint input_height,
							  char *output,
  							  const uint elm_size,
							  cudaMemcpyKind kind,
							  cudaStream_t stream)
{
	uint output_startx;
	uint output_starty;
	uint square_side_len;

	if (input_width >= input_height) //Usually the case
	{
		square_side_len = input_width;
		output_startx = 0;
		output_starty = (square_side_len - input_height) / 2;
	}
	else
	{
		square_side_len = input_height;
		output_startx = (square_side_len - input_width) / 2;
		output_starty = 0;
	}
	return embedded_frame_cpy(input,
	  					  	  input_width,
						  	  input_height,
	 					  	  output,
						  	  square_side_len,
			 			      square_side_len,
						  	  output_startx,
	  					  	  output_starty,
						  	  elm_size,
	  					  	  kind,
	  					  	  stream);
}

static __global__
void kernel_batched_embed_into_square(const char *input,
									  const uint input_width,
									  const uint input_height,
									  char *output,
									  const uint output_width,
									  const uint output_height,
									  const uint output_startx,
									  const uint output_starty,
									  const uint batch_size,
									  const uint elm_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint x = index % output_width;
	const uint y = index / output_width;

	if (index < output_width * output_height)
	{
		for (uint i = 0; i < batch_size; i++)
		{
			const uint batch_index = index + i * input_width * input_height * elm_size;

			if (x < output_startx || x >= output_startx + input_width
				|| y < output_starty || y >= output_starty + input_height)
				output[batch_index] = 0;
			else
			{
				if (output_startx == 0) // Horizontal black bands (top and bottom)
					output[batch_index] = input[batch_index - output_starty * input_width * elm_size];
				else // Vertical black bands (left and right)
					output[batch_index] = input[batch_index - (2 * y + 1) * output_startx * elm_size];
			}
		}
	}
}

void batched_embed_into_square(const char *input,
							const uint input_width,
							const uint input_height,
							char *output,
							const uint batch_size,
							const uint elm_size)
{
	uint output_startx;
	uint output_starty;
	uint square_side_len;

	if (input_width >= input_height) //Usually the case
	{
		square_side_len = input_width;
		output_startx = 0;
		output_starty = (square_side_len - input_height) / 2;
	}
	else
	{
		square_side_len = input_height;
		output_startx = (square_side_len - input_width) / 2;
		output_starty = 0;
	}

	size_t threads = get_max_threads_1d();
	size_t blocks = map_blocks_to_problem(square_side_len * square_side_len, threads);

	kernel_batched_embed_into_square<<<blocks, threads>>>(input,
		input_width,
		input_height,
		output,
		square_side_len,
		square_side_len,
		output_startx,
		output_starty,
		batch_size,
		elm_size);
	cudaCheckError();
}

cudaError_t crop_frame(const char *input,
					   const uint input_width,
					   const uint input_height,
					   const uint crop_start_x,
					   const uint crop_start_y,
					   const uint crop_width,
					   const uint crop_height,
					   char *output,
					   const uint elm_size,
					   cudaMemcpyKind kind,
					   cudaStream_t stream)
{
	assert(crop_start_x + crop_width <= input_width);
	assert(crop_start_y + crop_height <= input_height);

	const char *crop_start = input + elm_size * (crop_start_y * input_width + crop_start_x);
	return cudaMemcpy2DAsync(output,
				 		     crop_width * elm_size,
				   			 crop_start,
						     input_width * elm_size,
						     crop_width * elm_size,
						     crop_height,
						     kind,
						     stream);
}

cudaError_t crop_into_square(const char *input,
						     const uint input_width,
							 const uint input_height,
							 char *output,
							 const uint elm_size,
							 cudaMemcpyKind kind,
							 cudaStream_t stream)
{
	uint crop_start_x;
	uint crop_start_y;
	uint square_side_len;

	if (input_width >= input_height)
	{
		square_side_len = input_height;
		crop_start_x = (input_width - square_side_len) / 2;
		crop_start_y = 0;
	}
	else
	{
		square_side_len = input_width;
		crop_start_x = 0;
		crop_start_y = (input_height - square_side_len) / 2;
	}

	return crop_frame(input,
					  input_width,
					  input_height,
					  crop_start_x,
					  crop_start_y,
		  			  square_side_len,
					  square_side_len,
					  output,
					  elm_size,
					  kind,
					  stream);
}

static __global__
void kernel_batched_crop_into_square(const char *input,
									 const uint input_width,
									 const uint input_height,
									 const uint crop_start_x,
									 const uint crop_start_y,
									 const uint crop_width,
									 const uint crop_height,
									 char *output,
									 const uint elm_size,
									 const uint batch_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = index / crop_width;

	if (index < crop_width * crop_height)
	{
		for (uint i = 0; i < batch_size; i++)
		{
			const uint batch_index = index + i * input_width * input_height * elm_size;

			if (crop_start_x == 0) // Horizontal black bands (top and bottom)
				output[batch_index] = input[batch_index + crop_start_y * input_width * elm_size];
			else // Vertical black bands (left and right)
				output[batch_index] = input[batch_index + (2 * y + 1) * crop_start_x * elm_size];
		}
	}
}

void batched_crop_into_square(const char *input,
							  const uint input_width,
							  const uint input_height,
							  char *output,
							  const uint elm_size,
							  const uint batch_size)
{
	uint crop_start_x;
	uint crop_start_y;
	uint square_side_len;

	if (input_width >= input_height)
	{
		square_side_len = input_height;
		crop_start_x = (input_width - square_side_len) / 2;
		crop_start_y = 0;
	}
	else
	{
		square_side_len = input_width;
		crop_start_x = 0;
		crop_start_y = (input_height - square_side_len) / 2;
	}

	size_t threads = get_max_threads_1d();
	size_t blocks = map_blocks_to_problem(square_side_len * square_side_len, threads);

	kernel_batched_crop_into_square<<<blocks, threads>>>(input,
														 input_width,
														 input_height,
														 crop_start_x,
														 crop_start_y,
														 square_side_len,
														 square_side_len,
														 output,
														 elm_size,
														 batch_size);
	cudaCheckError();
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
		cudaXMemsetAsync(gpu_sum, 0, sizeof(float), stream);
	else
		return 0.f;

	// A SpanSize of 4 has been determined to be an optimal choice here.
	kernel_sum <4> << <blocks, threads, 0, stream >> >(
		input,
		gpu_sum,
		size);
	cudaCheckError();
	cudaXMemcpyAsync(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	cudaXFree(gpu_sum);

	return cpu_sum /= static_cast<float>(size);
}

static __global__
void kernel_average_complex_images(const cuComplex* in,
								   cuComplex* out,
								   size_t frame_res,
								   size_t nb_frames)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= frame_res)
		return;

	out[index].x = 0;
	out[index].y = 0;
	for (size_t i = 0; i < nb_frames; ++i)
	{
		cuComplex val = in[i * frame_res + index];
		out[index].x += hypotf(val.x, val.y);
	}
	out[index].x /= nb_frames;
	out[index].y /= nb_frames;
}

void average_complex_images(const cuComplex* in,
							cuComplex* out,
							size_t frame_res,
							size_t nb_frames)
{
	size_t threads = get_max_threads_1d();
	size_t blocks = map_blocks_to_problem(frame_res, threads);

	kernel_average_complex_images<<<blocks, threads>>>(in, out, frame_res, nb_frames);
	cudaCheckError();
	cudaStreamSynchronize(0);
	cudaCheckError();
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
		cudaXMemcpy(	resources->gpu_predecessor_,
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
	cudaXMemcpy(resources->gpu_predecessor_,
				cur,
				sizeof(cuComplex) * image_size,
				cudaMemcpyDeviceToDevice);

	/* Copying in order to later enqueue the (not summed up with values
	 * in gpu_unwrap_buffer_) phase image. */
	cudaXMemcpy(resources->gpu_angle_copy_,
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
	cudaXMemcpy(next_unwrap,
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
															1,
															middlex,
															middley,
															fd.width,
															fd.height,
															fd.frame_res());
	cudaCheckError();
	circ_shift_float << < blocks, threads, 0, stream >> > (	res->gpu_fy_,
															res->gpu_shift_fy_,
															1,
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

	kernel_add_complex_frames << < blocks, threads, 0, stream >> >(res->gpu_grad_eq_x_, res->gpu_grad_eq_y_, fd.frame_res());
	cudaCheckError();
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	kernel_unwrap2d_last_step << < blocks, threads, 0, stream >> > (output, res->gpu_grad_eq_x_, fd.frame_res());
	cudaCheckError();
}

__global__
void circ_shift(const cuComplex	*input,
				cuComplex	*output,
				const uint 	batch_size,
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

		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * size;

			const cuComplex rhs = input[batch_index];

			output[((width * shift_y) + shift_x) + i * size] = rhs;
		}
	}
}

__global__
void circ_shift_float(const float		*input,
					float		*output,
					const uint 	batch_size,
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

        for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * size;

			const float rhs = input[batch_index];

			output[((width * shift_y) + shift_x) + i * size] = rhs;
		}
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

float get_norm(const float	*matrix,
			   size_t		size)
{
	uint threads = 1024;
	uint blocks = map_blocks_to_problem(size, threads);

	holovibes::cuda_tools::UniquePtr<float> output(blocks);
	normalize_float_matrix<1024><<<blocks, threads, threads * sizeof(float)>>>(matrix, output, static_cast<uint>(size));

	float *intermediate_sum_cpu = new float[blocks]; //never more than 4096 threads
	//need to be on cpu for the sum
	cudaXMemcpy(intermediate_sum_cpu, output, blocks * sizeof(float), cudaMemcpyDeviceToHost); //TODO faire la somme sur GPU
	float sum = 0;
	for (uint i = 0; i < blocks; i++)
		sum += intermediate_sum_cpu[i];
	delete[] intermediate_sum_cpu;

	return sum;
}
