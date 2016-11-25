#include <device_launch_parameters.h>
#include <cmath>
# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */

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
    local_adjust = -2.f * pi;
  else if (local_diff < -pi)
    local_adjust = 2.f * pi;
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
		fx[index] = (i - static_cast<float>(lrintf(static_cast<float>(width) / 2)));
		fy[index] = (j - static_cast<float>(lrintf(static_cast<float>(height) / 2)));

		/*z init*/
		z[index].x = cosf(input[index]);
		z[index].y = sinf(input[index]);
	}
}

__global__ void kernel_init_unwrap_2d_complex(
	unsigned int width,
	unsigned int height,
	unsigned int frame_res,
	cufftComplex *input,
	float *fx,
	float *fy,
	cufftComplex *z)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = j * blockDim.x * gridDim.x + i;

	if (index < frame_res)
	{
		/*fx and fy init*/

		fx[index] = (i - static_cast<float>(lrintf(static_cast<float>(width) / 2)));
		fy[index] = (j - static_cast<float>(lrintf(static_cast<float>(height) / 2)));

		/*z init*/
		const float modulus = sqrtf(abs(input[index].x * input[index].x + input[index].y * input[index].y));
		if (modulus == 0)
		{
			z[index].x = 0;
			z[index].y = 0;
		}
		else
		{
			z[index].x = input[index].x / modulus;
			z[index].y = input[index].y / modulus;
		}
	}
}