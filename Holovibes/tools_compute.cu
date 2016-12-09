#include <device_launch_parameters.h>

#include "tools_compute.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_complex_divide(
  cufftComplex* image,
  const unsigned int size,
  const float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_float_divide(
  float* input,
  const unsigned int size,
  const float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    input[index] /= divider;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_multiply_frames_complex(
	const cufftComplex* input1,
	const cufftComplex* input2,
	cufftComplex* output,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index].x = input1[index].x * input2[index].x;
		output[index].y = input1[index].y * input2[index].y;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_multiply_frames_float(
	const float* input1,
	const float* input2,
	float* output,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index] = input1[index] * input2[index];
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_substract_ref(
	cufftComplex*      input,
	cufftComplex*      reference,
	const unsigned int size,
	const unsigned int frame_size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < size)
	{
		input[index].x -= reference[index % frame_size].x;
		index += blockDim.x * gridDim.x;
	}
}

void substract_ref(
	cufftComplex* input,
	cufftComplex* reference,
	const unsigned int frame_resolution,
	const unsigned int nframes,
	cudaStream_t stream)
{
	const unsigned int n_frame_resolution = frame_resolution * nframes;
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);
    kernel_substract_ref << <blocks, threads, 0, stream >> >(input, reference, n_frame_resolution, frame_resolution);
}

__global__ void kernel_mean_images(
	cufftComplex *input,
	cufftComplex *output,
	unsigned int n,
	unsigned int frame_size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
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

void mean_images(
	 cufftComplex* input,
	 cufftComplex* output,
	unsigned int n,
	unsigned int frame_size,
	cudaStream_t stream)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_size, threads);

	kernel_mean_images << <blocks, threads, 0, stream >> >(input, output, n, frame_size);
}

__global__ void kernel_multiply_complexes_by_floats_(
	const float* input1,
	const float* input2,
	cufftComplex* output1,
	cufftComplex* output2,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output1[index].x = output1[index].x * input1[index];
		output1[index].y = output1[index].y * input1[index];
		output2[index].x = output2[index].x * input2[index];
		output2[index].y = output2[index].y * input2[index];
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_multiply_complexes_by_single_complex(
	cufftComplex* output1,
	cufftComplex* output2,
	const cufftComplex input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		const cufftComplex cpy_o1 = output1[index];
		const cufftComplex cpy_o2 = output2[index];

		output1[index].x = cpy_o1.x * input.x - cpy_o1.y * input.y;
		output1[index].y = cpy_o1.x * input.y + cpy_o1.y * input.x;
		output2[index].x = cpy_o2.x * input.x - cpy_o2.y * input.y;
		output2[index].y = cpy_o2.x * input.y + cpy_o2.y * input.x;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_multiply_complex_by_single_complex(
	cufftComplex* output,
	const cufftComplex input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		const cufftComplex cpy_o1 = output[index];

		output[index].x = cpy_o1.x * input.x - cpy_o1.y * input.y;
		output[index].y = cpy_o1.x * input.y + cpy_o1.y * input.x;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_conjugate_complex(
	cufftComplex* output,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index].y = -output[index].y;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_multiply_complex_frames_by_complex_frame(
	cufftComplex* output1,
	cufftComplex* output2,
	const cufftComplex* input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		const cufftComplex cpy_o1 = output1[index];
		const cufftComplex cpy_o2 = output2[index];

		output1[index].x = cpy_o1.x * input[index].x - cpy_o1.y * input[index].y;
		output1[index].y = cpy_o1.x * input[index].y + cpy_o1.y * input[index].x;
		output2[index].x = cpy_o2.x * input[index].x - cpy_o2.y * input[index].y;
		output2[index].y = cpy_o2.x * input[index].y + cpy_o2.y * input[index].x;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_norm_ratio(
	const float* input1,
	const float* input2,
	cufftComplex* output1,
	cufftComplex* output2,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		float norm = input1[index] * input1[index] + input2[index] * input2[index];

		if (norm != 0)
		{
			float coeff_x = input1[index] / norm;
			float coeff_y = input2[index] / norm;

			output1[index].x = output1[index].x * coeff_x;
			output1[index].y = output1[index].y * coeff_x;
			output2[index].x = output2[index].x * coeff_y;
			output2[index].y = output2[index].y * coeff_y;
		}
		else
		{
			output1[index].x = 0;
			output1[index].y = 0;
			output2[index].x = 0;
			output2[index].y = 0;
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_add_complex_frames(
	cufftComplex* output,
	const cufftComplex* input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index].x += input[index].x;
		output[index].y += input[index].y;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_unwrap2d_last_step(
	float* output,
	const cufftComplex* input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index] = input[index].y / (-2 * M_PI);
		index += blockDim.x * gridDim.x;
	}
}



__global__ void kernel_convergence(
	cufftComplex* input1,
	cufftComplex* input2)
{
	input1[0].x = 0;
	input1[0].y = 0;
	input2[0].x = 0;
	input2[0].y = 0;
}