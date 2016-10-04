#include <device_launch_parameters.h>

#include "convolution.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_multiply_kernel(
	cufftComplex* input,
	cufftComplex* gpu_special_queue,
	const unsigned int gpu_special_queue_buffer_length,
	const unsigned int frame_resolution,
	const unsigned int i_width,
	const float* kernel,
	const unsigned int k_width,
	const unsigned int k_height,
	const unsigned int nsamples,
	const unsigned int start_index,
	const unsigned int max_index )
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int n, m, z;
	unsigned int size = frame_resolution * nsamples;
	unsigned int k_size = k_width * k_height;
	while (index < frame_resolution)
	{
		cufftComplex sum = make_cuComplex(0, 0);

		for (z = 0; z < nsamples; ++z)
		for (m = 0; m < k_width; ++m)
		for (n = 0; n < k_height; ++n) {
			cufftComplex a = gpu_special_queue[(index + m + n * i_width + (((z + start_index) % max_index) * frame_resolution)) % gpu_special_queue_buffer_length];
			float b = kernel[m + n * k_width + (z * k_size)];
			sum.x += a.x * b;// - a.y * b.y;
			sum.y += a.y * b;// + a.x * b.y;
			sum.x /= nsamples * k_width * k_height;
		}
		input[index] = sum;
		index += blockDim.x * gridDim.x;
	}
}

void convolution_kernel(
	cufftComplex* input,
	cufftComplex* gpu_special_queue,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const float* kernel,
	const unsigned int k_width,
	const unsigned int k_height,
	const unsigned int k_z,
	unsigned int& gpu_special_queue_start_index,
	const unsigned int& gpu_special_queue_max_index,
	cudaStream_t stream)
{
	// const unsigned int n_frame_resolution = frame_resolution * nframes;

	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);

	if (gpu_special_queue_start_index == 0)
		gpu_special_queue_start_index = gpu_special_queue_max_index - 1;
	else
		--gpu_special_queue_start_index;
	cudaMemcpy(
		gpu_special_queue + frame_resolution * gpu_special_queue_start_index,
		input,
		sizeof(cufftComplex)* frame_resolution,
		cudaMemcpyDeviceToDevice);
	
	unsigned int gpu_special_queue_buffer_length = gpu_special_queue_max_index * frame_resolution;

	kernel_multiply_kernel<<<blocks, threads, 0, stream>>>(
		input,
        gpu_special_queue,
		gpu_special_queue_buffer_length,
		frame_resolution,
		frame_width,
		kernel,
		k_width,
		k_height,
		k_z,
		gpu_special_queue_start_index,
		gpu_special_queue_max_index
		);
		
	cudaStreamSynchronize(stream);
}