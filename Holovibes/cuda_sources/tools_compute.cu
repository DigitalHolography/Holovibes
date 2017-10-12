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
void kernel_float_divide(float		*input,
						const uint	size,
						const float	divider)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  //while (index < size)
  {
    input[index] /= divider;
    //index += blockDim.x * gridDim.x;
  }
}

__global__
void kernel_multiply_frames_complex(const cuComplex	*input1,
									const cuComplex	*input2,
									cuComplex		*output,
									const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < size)
	{
		output[index].x = input1[index].x * input2[index].x;
		output[index].y = input1[index].y * input2[index].y;
		//index += blockDim.x * gridDim.x;
	}
}

__global__
void kernel_multiply_frames_float(const float	*input1,
								const float		*input2,
								float			*output,
								const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < size)
	{
		output[index] = input1[index] * input2[index];
		//index += blockDim.x * gridDim.x;
	}
}

__global__
void kernel_substract_ref(cuComplex	*input,
						cuComplex	*reference,
						const uint	size,
						const uint	frame_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < size)
	{
		input[index].x -= reference[index % frame_size].x;
		//index += blockDim.x * gridDim.x;
	}
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
}

__global__
void kernel_mean_images(cuComplex	*input,
						cuComplex	*output,
						uint		n,
						uint		frame_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	//while (index < frame_size)
	{
		float tmp = 0;
		for (int i = 0; i < n; i++)
			tmp += input[index + i * frame_size].x;
		tmp /= n;
		output[index].x = tmp;
		//index += blockDim.x * gridDim.x;
	}
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
							uint	block_size)
{
	struct extr_index min;
	struct extr_index max;
	for (uint i = 1; i < block_size; i++)
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
	global_extremums << <1, 1, 0, 0 >> > (local_extr,
		global_extr,
		block_size);

	struct extr_index extremum[2];
	cudaMemcpy(extremum, global_extr, sizeof(struct extr_index) * 2, cudaMemcpyDeviceToHost);

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
