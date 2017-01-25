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