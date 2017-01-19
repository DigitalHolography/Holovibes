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
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_complex_divide(complex		*image,
									  const uint	size,
									  const float	divider)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_float_divide(float		*input,
									const uint	size,
									const float	divider)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    input[index] /= divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_multiply_frames_complex(	const complex	*input1,
												const complex	*input2,
												complex			*output,
												const uint		size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index].x = input1[index].x * input2[index].x;
		output[index].y = input1[index].y * input2[index].y;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_multiply_frames_float(	const float	*input1,
												const float	*input2,
												float		*output,
												const uint	size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index] = input1[index] * input2[index];
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_substract_ref(	complex		*input,
										complex		*reference,
										const uint	size,
										const uint	frame_size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		input[index].x -= reference[index % frame_size].x;
		index += blockDim.x * gridDim.x;
	}
}

void substract_ref(	complex			*input,
					complex			*reference,
					const uint		frame_resolution,
					const uint		nframes,
					cudaStream_t	stream)
{
	const uint	n_frame_resolution = frame_resolution * nframes;
	uint		threads = get_max_threads_1d();
	uint		blocks = map_blocks_to_problem(frame_resolution, threads);
    kernel_substract_ref << <blocks, threads, 0, stream >> >(input, reference, n_frame_resolution, frame_resolution);
}

__global__ void kernel_mean_images(	complex		*input,
									complex		*output,
									uint		n,
									uint		frame_size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < frame_size)
	{
		float tmp = 0;
		for (int i = 0; i < n; i++)
			tmp += input[index + i * frame_size].x;
		tmp /= n;
		output[index].x = tmp;
		index += blockDim.x * gridDim.x;
	}
}

void mean_images(	complex			*input,
					complex			*output,
					uint			n,
					uint			frame_size,
					cudaStream_t	stream)
{
	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_size, threads);

	kernel_mean_images << <blocks, threads, 0, stream >> >(input, output, n, frame_size);
}