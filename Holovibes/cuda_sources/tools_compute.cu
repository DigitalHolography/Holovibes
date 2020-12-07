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

#include "reduce.cuh"
#include "map.cuh"

#include <stdio.h>

#define AUTO_CONTRAST_COMPENSATOR 10000

__global__
void kernel_complex_divide(cuComplex	*image,
						 const uint		frame_res,
						 const float	divider,
						 const uint 	batch_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < frame_res)
	{
		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * frame_res;

			image[batch_index].x /= divider;
			image[batch_index].y /= divider;
		}
	}
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

void gpu_normalize(float* const input,
                   double* const result_reduce,
                   const uint frame_res,
                   const uint norm_constant,
                   cudaStream_t stream)
{
    gpu_reduce(input, result_reduce, frame_res);

    // Let x be a pixel, after renormalization
    // x = x * 2^(norm_constant) / mean
    // x = x * 2^(norm_constant) * frame_res / reduce_result
    const double multiplier = (1 << norm_constant);
    auto map_function = [multiplier, frame_res, result_reduce] __device__ (const float input_pixel)
        {
            // This operation needs to be computed on double to avoid overflow
            return static_cast<float>(static_cast<double>(input_pixel * multiplier * frame_res) / (*result_reduce));
        };

    map_generic(input, input, frame_res, map_function, stream);
}