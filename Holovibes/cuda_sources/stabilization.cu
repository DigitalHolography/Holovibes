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

#include "stabilization.cuh"
#include "common.cuh"
#include "cuda_tools/unique_ptr.hh"


struct rect
{
	int x;
	int y;
	int w;
	int h;
};

struct point
{
	int x;
	int y;
};

__global__
void kernel_extract_frame(const float	*input,
						float			*output,
						const uint		input_w,
						const struct rect	frame)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / frame.w;
	if (new_y < frame.h)
	{
		const uint new_x = index % frame.w;
		const uint old_x = new_x + frame.x;
		const uint old_y = new_y + frame.y;

		const uint old_index = old_y * input_w + old_x;
		output[index] = input[old_index];
	}
}

void extract_frame(const float	*input,
				float			*output,
				const uint		input_w,
				const holovibes::gui::Rectangle&	frame)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame.area(), threads);
	struct rect rect_frame = { frame.x(), frame.y(), frame.width(), frame.height() };
	kernel_extract_frame << <blocks, threads, 0, 0 >> > (input, output, input_w, rect_frame);
	cudaStreamSynchronize(0);
}

__global__
void kernel_resize(const float	*input,
					float		*output,
					const struct point old_size,
					const struct point new_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / new_size.x;
	if (new_y < new_size.y)
	{
		const uint new_x = index % new_size.x;
		const uint old_x = new_x / (new_size.x / old_size.x);
		const uint old_y = new_y / (new_size.y / old_size.y);

		const uint old_index = old_y * old_size.x + old_x;
		output[index] = input[old_index];
	}
}


void gpu_resize(const float		*input,
				float			*output,
				QPoint			old_size,
				QPoint			new_size,
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(new_size.x() * new_size.y(), threads);
	struct point old_s = { old_size.x(), old_size.y() };
	struct point new_s = { new_size.x(), new_size.y() };
	kernel_resize << <blocks, threads, 0, stream >> > (input, output, old_s, new_s);
	cudaStreamSynchronize(0);
}


__global__
void kernel_rotation_180(float			*frame,
							point		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / size.x;
	if (new_y < size.y / 2)
	{
		const uint new_x = index % size.y;
		const uint old_y = size.y - new_y - 1;
		const uint old_x = size.x - new_x - 1;
		const uint old_index = old_y * size.x + old_x;
		float tmp = frame[old_index];
		frame[old_index] = frame[index];
		frame[index] = tmp;
	}
}

void rotation_180(float			*frame,
					QPoint		size,
					cudaStream_t stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_rotation_180 << <blocks, threads, 0, stream >> > (frame, s);
}






__global__
void kernel_sum_columns_inplace(float		*input,
								point		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size.y)
	{
		for (uint i = 1; i < size.x; ++i)
			input[i * size.x + index] += input[(i - 1) * size.x + index];
	}
}


__global__
void kernel_sum_lines_inplace_squared(float		*input,
									point		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size.x)
	{
		const uint start = index * size.x;
		input[start] *= input[start];
		for (uint i = 1; i < size.y; ++i)
			input[start + i] = input[start + i] * input[start + i] + input[start + i - 1];
	}
}

void sum_inplace_squared(float			*input,
						QPoint			size,
						cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_sum_lines_inplace_squared << <blocks, threads, 0, stream >> > (input, s);
	cudaStreamSynchronize(stream);
	kernel_sum_columns_inplace << <blocks, threads, 0, stream >> > (input, s);
}


__global__
void kernel_sum_lines_inplace(float		*input,
							point		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size.x)
	{
		const uint start = index * size.x;
		for (uint i = 1; i < size.y; ++i)
			input[start + i] += input[start + i - 1];
	}
}
void sum_left_right_inplace(float			*input,
							QPoint			size,
							cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_sum_lines_inplace << <blocks, threads, 0, stream >> > (input, s);
	cudaStreamSynchronize(stream);
	kernel_sum_columns_inplace << <blocks, threads, 0, stream >> > (input, s);
}

__global__
void kernel_sum_lines(const float	*input,
					float			*output,
					point			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size.x)
	{
		const uint start = index * size.x;
		output[start] = input[start];
		for (uint i = 1; i < size.y; ++i)
			output[start + i] = input[start + i] + output[start + i - 1];
	}
}

void sum_left_right(const float	*input,
					float		*output,
					QPoint		size,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_sum_lines << <blocks, threads, 0, stream >> > (input, output, s);
	cudaStreamSynchronize(stream);
	kernel_sum_columns_inplace << <blocks, threads, 0, stream >> > (output, s);
}


__global__
void kernel_compute_numerator(const float	*sum_a,
						const float			*sum_b,
						float				*sum_convolution,
						point				size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = index / size.x;
	if (y < size.y)
	{
		const uint x = index % size.x;
		sum_convolution[index] -= (sum_a[index] * sum_b[index]) / (x + y + 1);
	}
}

void compute_numerator(const float	*sum_a,
					const float		*sum_b,
					float			*sum_convolution,
					QPoint			size,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_compute_numerator << <blocks, threads, 0, stream >> > (sum_a, sum_b, sum_convolution, s);
}


__global__
void k_sum_squared_minus_square_sum(
					float			*matrix,
					const float		*sum_squared,
					point			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = index / size.x;
	if (y < size.y)
	{
		const uint x = index % size.x;
		matrix[index] = sum_squared[index] - (matrix[index] * matrix[index]) / (x + y + 1);
		if (matrix[index] < 0)
			matrix[index] = 0;
	}
}

void sum_squared_minus_square_sum(
					float			*matrix,
					const float		*sum_squared,
					QPoint			size,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	k_sum_squared_minus_square_sum << <blocks, threads, 0, stream >> > (matrix, sum_squared, s);
}

// see: https://en.wikipedia.org/wiki/Fast_inverse_square_root
__device__
float fast_invert_sqrt(float x)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = x * 0.5F;
	y = x;
	i = *(long *)&y;                       // evil floating point bit level hacking
	i = 0x5f3759df - (i >> 1);               // what the fuck? 
	y = *(float *)&i;
	y = y * (threehalfs - (x2 * y * y));   // 1st iteration

	return y;
}

__global__
void kernel_correlation(float			*numerator,
						const float		*denominator1,
						const float		*denominator2,
						point			size)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int ignored = size.x / 20;
	const int y = index / size.x;
	if (y >= size.y)
		return;
	if (y < ignored || y >= size.y - ignored)
		numerator[index] = 0;
	else
	{
		const int x = index % size.x;
		if (x < ignored || x > size.x - ignored)
			numerator[index] = 0;
		else
			numerator[index] *= fast_invert_sqrt(denominator1[index] * denominator2[index]);
	}
}


void correlation(float			*numerator,
				const float		*denominator1,
				const float		*denominator2,
				QPoint			size,
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	const struct point s = { size.x(), size.y() };
	kernel_correlation << <blocks, threads, 0, stream >> > (numerator, denominator1, denominator2, s);
}

__global__
void kernel_pad_frame(const float	*input,
						float		*output,
						point		old_size,
						point		new_size)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = index / new_size.x;
	if (y < new_size.y)
	{
		const int x = index % new_size.x;
		if (x >= old_size.x || y >= old_size.y)
			output[index] = 0;
		else
		{
			const int old_x = x * old_size.x / new_size.x;
			const int old_y = y * old_size.y / new_size.y;
			output[index] = input[old_y * old_size.x + old_x];
		}
	}
}

void pad_frame(float		*frame,
				QPoint		old_size,
				QPoint		new_size,
				cudaStream_t	stream)
{
	cuda_tools::UniquePtr<float> tmp(new_size.x() * new_size.y());
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(new_size.x() * new_size.y(), threads);
	const struct point s = { new_size.x(), new_size.y() };
	const struct point old_s = { old_size.x(), old_size.y() };
	kernel_pad_frame << <blocks, threads, 0, stream >> > (frame, tmp.get(), old_s, s);
	cudaStreamSynchronize(0);
	cudaMemcpy(frame, tmp.get(), new_size.x() * new_size.y() * sizeof(float), cudaMemcpyDeviceToDevice);
}






