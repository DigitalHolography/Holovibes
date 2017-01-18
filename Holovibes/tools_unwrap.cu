#include <device_launch_parameters.h>
#include <cmath>

#include "tools_unwrap.cuh"

__global__ void kernel_extract_angle(
  const cufftComplex* input,
  float* output,
  const size_t size)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size)
    return;

  // We use std::atan2 in order to obtain results in [-pi; pi].
  output[index] = std::atan2(input[index].y, input[index].x);
}

__global__ void kernel_unwrap(
  float* pred,
  float* cur,
  float* output,
  const size_t size)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size)
    return;
  const float pi = M_PI;

  float local_diff = cur[index] - pred[index];
  // Unwrapping //
  float local_adjust;
  if (local_diff > pi)
    local_adjust = -M_2PI;
  else if (local_diff < -pi)
    local_adjust = M_2PI;
  else
    local_adjust = 0.f;

  // Cumulating each angle with its correction
  output[index] = cur[index] + local_adjust;
}

__global__ void kernel_compute_angle_mult(
  const cufftComplex* pred,
  const cufftComplex* cur,
  float* output,
  const size_t size)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size)
    return;

  cufftComplex conj_prod;
  conj_prod = cur[index];

  conj_prod.x *= pred[index].x;
  conj_prod.x += cur[index].y * pred[index].y;

  conj_prod.y *= pred[index].x;
  conj_prod.y -= cur[index].x * pred[index].y;

  output[index] = std::atan2(conj_prod.y, conj_prod.x);
}

__global__ void kernel_compute_angle_diff(
  const cufftComplex* pred,
  const cufftComplex* cur,
  float* output,
  const size_t size)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size)
    return;

  cufftComplex diff;
  diff = cur[index];
  diff.x -= pred[index].x;
  diff.y -= pred[index].y;

  output[index] = std::atan2(diff.y, diff.x);
}

__global__ void kernel_correct_angles(
  float* data,
  const float* corrections,
  const size_t image_size,
  const size_t history_size)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= image_size)
    return;

  for (auto correction_idx = index;
    correction_idx < history_size * image_size;
    correction_idx += image_size)
  {
    data[index] += corrections[correction_idx];
  }
}

__global__ void kernel_init_unwrap_2d(
	unsigned int width,
	unsigned int height,
	unsigned int frame_res,
	float *input,
	float *fx,
	float *fy,
	cufftComplex *z)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = j * blockDim.x * gridDim.x + i;

	if (index < frame_res)
	{
		fx[index] = (i - static_cast<float>(lrintf(static_cast<float>((width) >> 1))));
		fy[index] = (j - static_cast<float>(lrintf(static_cast<float>((height) >> 1))));

		/*z init*/
		z[index].x = cosf(input[index]);
		z[index].y = sinf(input[index]);
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
		output[index] = input[index].y / -M_2PI;
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