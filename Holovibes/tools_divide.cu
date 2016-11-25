#include <device_launch_parameters.h>

#include "tools_divide.cuh"
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

__global__ void kernel_substract_ref_8(
	cufftComplex*      input,
	unsigned char*    reference,
	const unsigned int size,
	const unsigned int frame_size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < size)
	{
		input[index].x -= reference[index % frame_size];
		index += blockDim.x * gridDim.x;
	}
}

__global__ void kernel_substract_ref_16(
	cufftComplex*      input,
	unsigned short*    reference,
	const unsigned int size,
	const unsigned int frame_size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < size)
	{
		input[index].x -= (reference[index % frame_size] * 257);
		index += blockDim.x * gridDim.x;
	}
}


void substract_ref(
	cufftComplex* input,
	void*         reference,
	const unsigned int pixel_size,
	const unsigned int frame_resolution,
	const unsigned int nframes,
	cudaStream_t stream)
{
	const unsigned int n_frame_resolution = frame_resolution * nframes;
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);
	if (pixel_size == 1)
    kernel_substract_ref_8 << <blocks, threads, 0, stream >> >(input, static_cast<unsigned char*>(reference), n_frame_resolution, frame_resolution);
	if (pixel_size == 2)
	kernel_substract_ref_16 << <blocks, threads, 0, stream >> >(input, static_cast<unsigned short*>(reference), n_frame_resolution, frame_resolution);
}