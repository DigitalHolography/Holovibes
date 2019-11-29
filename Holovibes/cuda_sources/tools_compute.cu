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

#include "tools_compute.cuh"
#include "tools_unwrap.cuh"
#include "min_max.cuh"

#include <stdio.h>

#define AUTO_CONTRAST_COMPENSATOR 10000

__global__
void kernel_complex_divide(cuComplex	*image,
						 const uint		size,
						 const float	divider)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  //while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    //index += blockDim.x * gridDim.x;
  }
}


__global__
void kernel_real_part_divide(cuComplex	*image,
	const uint		size,
	const float	divider)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		image[index].x = image[index].x / divider * AUTO_CONTRAST_COMPENSATOR;
}

void gpu_real_part_divide(cuComplex	*image,
	const uint	size,
	const float	divider)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(size, threads);

	kernel_real_part_divide <<< blocks, threads >>>(image, size, divider);
	cudaCheckError();
}



__global__
void kernel_float_divide(float		*input,
						const uint	size,
						const float	divider)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    input[index] /= divider;
}

__global__
void kernel_multiply_frames_complex(const cuComplex	*input1,
									const cuComplex	*input2,
									cuComplex		*output,
									const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		const float new_x = (input1[index].x * input2[index].x) - (input1[index].y * input2[index].y);
		const float new_y = (input1[index].y * input2[index].x) + (input1[index].x * input2[index].y);
		output[index].x = new_x;
		output[index].y = new_y;
	}
}

__global__
void kernel_divide_frames_float(const float	*numerator,
								const float	*denominator,
								float		*output,
								const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		const float new_x = numerator[index] / denominator[index];
		output[index] = new_x;
	}
}

void multiply_frames_complex(const cuComplex	*input1,
								const cuComplex	*input2,
								cuComplex		*output,
								const uint		size,
								cudaStream_t	stream)
{
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(size, threads);
	kernel_multiply_frames_complex << <blocks, threads, 0, stream >> > (input1, input2, output, size);
	cudaCheckError();
}

__global__
void kernel_multiply_frames_float(const float	*input1,
								const float		*input2,
								float			*output,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	output[index] = input1[index] * input2[index];
}

__global__
void kernel_substract_ref(cuComplex	*input,
						cuComplex	*reference,
						const uint	size,
						const uint	frame_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	input[index].x -= reference[index % frame_size].x;
}

void substract_ref(cuComplex	*input,
				cuComplex		*reference,
				const uint		frame_resolution,
				const uint		nframes,
				cudaStream_t	stream)
{
	const uint	n_frame_resolution = frame_resolution * nframes;
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_resolution, threads);
    kernel_substract_ref << <blocks, threads, 0, stream >> >(input, reference, n_frame_resolution, frame_resolution);
	cudaCheckError();
}

static __global__
void kernel_subtract_frame_complex(cuComplex* img1,
	cuComplex* img2,
	cuComplex* out,
	size_t frame_res)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= frame_res)
		return;

	out[index].x = img1[index].x - img2[index].x;
	out[index].y = img1[index].y - img2[index].y;
}

void subtract_frame_complex(cuComplex* img1,
	cuComplex* img2,
	cuComplex* out,
	size_t frame_res,
	cudaStream_t stream)
{
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_res, threads);
	kernel_subtract_frame_complex <<<blocks, threads, 0, stream>>>(img1, img2, out, frame_res);
	cudaCheckError();
	cudaStreamSynchronize(stream);
}

__global__
void kernel_mean_images(cuComplex	*input,
						cuComplex	*output,
						uint		n,
						uint		frame_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	float tmp = 0;
	for (int i = 0; i < n; i++)
		tmp += input[index + i * frame_size].x;
	tmp /= n;
	output[index].x = tmp;
}

void mean_images(cuComplex		*input,
				cuComplex		*output,
				uint			n,
				uint			frame_size,
				cudaStream_t	stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_size, threads);

	kernel_mean_images << <blocks, threads, 0, stream >> >(input, output, n, frame_size);
	cudaCheckError();
}


struct extr_index
{
	float extr;
	uint index;
};

static __global__
void local_extremums(float	*input,
							struct extr_index	*output,
							uint	block_size,
							uint	size)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	uint begin = id * block_size;
	if (begin < size)
	{
		uint min_id = begin;
		uint max_id = begin;
		uint end = begin + block_size;
		if (end > size)
			end = size;
		float min = input[begin];
		float max = input[begin];
		while (++begin < end)
		{
			float elt = input[begin];
			if (elt < min)
			{
				min = elt;
				min_id = begin;
			}
			else if (elt > max)
			{
				max_id = begin;
				max = elt;
			}
		}
		output[2 * id].index = min_id;
		output[2 * id].extr = min;
		output[2 * id + 1].index = max_id;
		output[2 * id + 1].extr = max;
	}
}

static __global__
void global_extremums(struct extr_index	*input,
							struct extr_index	*output,
							uint	nb_blocks)
{
	struct extr_index min = input[0];
	struct extr_index max = input[1];
	for (uint i = 1; i < nb_blocks; i++)
	{
		struct extr_index current_min = input[2 * i];
		struct extr_index current_max = input[2 * i + 1];
		if (current_min.extr < min.extr)
			min = current_min;
		else if (current_max.extr > max.extr)
			max = current_max;
	}
	output[0] = min;
	output[1] = max;
}


void gpu_extremums(float			*input,
					const uint		size,
					float			*min,
					float			*max,
					uint			*min_index,
					uint			*max_index,
					cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const ushort block_size = 1024;
	const ushort nb_blocks = size / block_size + 1;

	struct extr_index *local_extr = nullptr;
	struct extr_index *global_extr = nullptr;
	cudaMalloc(&local_extr, sizeof(struct extr_index) * 2 * nb_blocks);
	cudaMalloc(&global_extr, sizeof(struct extr_index) * 2);

	uint blocks = map_blocks_to_problem(nb_blocks, threads);
	local_extremums << <threads, blocks, 0, 0 >> > (input,
		local_extr,
		block_size,
		size);
	cudaCheckError();
	global_extremums << <1, 1, 0, 0 >> > (local_extr,
		global_extr,
		nb_blocks);
	cudaCheckError();

	struct extr_index extremum[2];
	cudaMemcpy(extremum, global_extr, sizeof(struct extr_index) * 2, cudaMemcpyDeviceToHost);
	cudaCheckError();

	cudaFree(local_extr);
	cudaFree(global_extr);

	if (min)
		*min = extremum[0].extr;
	if (max)
		*max = extremum[1].extr;

	if (min_index)
		*min_index = extremum[0].index;
	if (max_index)
		*max_index = extremum[1].index;

}

__global__
void kernel_substract_const(float		*frame,
							uint		frame_size,
							float		x)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_size)
		frame[id] -= x;
}

void gpu_substract_const(float		*frame,
						uint		frame_size,
						float		x)
{
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_size, threads);
	kernel_substract_const << <blocks, threads, 0, 0 >> >(frame, frame_size, x);
	cudaCheckError();
}

__global__
void kernel_multiply_const(float		*frame,
							uint		frame_size,
							float		x)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_size)
		frame[id] *= x;
}

void gpu_multiply_const(float		*frame,
						uint		frame_size,
						float		x)
{
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_size, threads);
	kernel_multiply_const << <blocks, threads, 0, 0 >> >(frame, frame_size, x);
	cudaCheckError();
}

void gpu_multiply_const(cuComplex * frame, uint frame_size, cuComplex x)
{
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_size, threads);
	kernel_multiply_complex_by_single_complex << <blocks, threads, 0, 0 >> >(frame, x, frame_size);
	cudaCheckError();
}

void normalize_frame(float* frame, uint frame_res)
{
	float min, max;

	get_minimum_maximum_in_image(frame, frame_res, &min, &max);

	gpu_substract_const(frame, frame_res, min);
	cudaCheckError();
	gpu_multiply_const(frame, frame_res, 1 / (max - min));
	cudaStreamSynchronize(0);
	cudaCheckError();
}
