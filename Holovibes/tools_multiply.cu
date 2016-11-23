#include <device_launch_parameters.h>

#include "tools_multiply.cuh"
#include "hardware_limits.hh"

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
		float norm = sqrtf(input1[index] * input1[index] + input2[index] * input2[index]);
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

__global__ void kernel_phi(
	float* output,
	const cufftComplex* input,
	const cufftComplex coeff,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		output[index] = input[index].y / coeff.y;
		index += blockDim.x * gridDim.x;
	}
}