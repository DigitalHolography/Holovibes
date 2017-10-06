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

#include <numeric>
#include "contrast_correction.cuh"

static __global__
void apply_contrast(float		*input,
					const uint	size,
					const float	factor,
					const float	min)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
	{
		input[index] = factor * (input[index] - min);
	}
}

void manual_contrast_correction(float			*input,
								const uint		size,
								const ushort	dynamic_range,
								const float		min,
								const float		max,
								cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	const float factor = dynamic_range / (max - min + FLT_EPSILON);
	apply_contrast << <blocks, threads, 0, stream >> > (input, size, factor, min);
}

static __global__
void local_extremums(float	*input,
					float	*output,
					uint	block_size,
					uint	size)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	uint begin = id * block_size;
	if (begin < size)
	{
		uint end = begin + block_size;
		if (end > size)
			end = size;
		float min = input[begin];
		float max = input[begin];
		while (++begin < end)
		{
			float elt = input[begin];
			if (elt < min)
				min = elt;
			if (elt > max)
				max = elt;
		}
		output[2 * id] = min;
		output[2 * id + 1] = max;
	}
}

static __global__
void global_extremums(float	*input,
					float	*output,
					uint	block_size)
{
	float min = *input;
	float max = *input;
	for (uint i = 1; i < block_size; i++)
	{
		float min_elt = input[2 * i];
		float max_elt = input[2 * i + 1];
		if (min_elt < min)
			min = min_elt;
		else if (max_elt > max)
			max = max_elt;
	}
	output[0] = min;
	output[1] = max;
}



void auto_contrast_correction(float			*input,
							const uint		size,
							const uint		offset,
							float			*min,
							float			*max,
							cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const ushort block_size = 1024;
	const ushort nb_blocks = size / block_size + 1;

	float *local_extr = nullptr;
	float *global_extr = nullptr;
	cudaMalloc(&local_extr, sizeof(float) * 2 * nb_blocks);
	cudaMalloc(&global_extr, sizeof(float) * 2);

	uint blocks = map_blocks_to_problem(nb_blocks, threads);
	local_extremums << <threads, blocks, 0, 0 >> > (input,
		local_extr,
		block_size,
		size);
	global_extremums << <1, 1, 0, 0 >> > (local_extr,
		global_extr,
		block_size);

	float extremum[2];
	cudaMemcpy(extremum, global_extr, sizeof(float) * 2, cudaMemcpyDeviceToHost);

	cudaFree(local_extr);
	cudaFree(global_extr);

	*min = extremum[0];
	*max = extremum[1];

	*min = ((*min < 1.0f) ? (1.0f) : (*min));
	*max = ((*max < 1.0f) ? (1.0f) : (*max));
}
