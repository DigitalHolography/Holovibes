#include <device_launch_parameters.h>

#include "flowgraphy.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_flowgraphy(
	cufftComplex* input,
	const cufftComplex* gpu_special_queue,
	const unsigned int gpu_special_queue_buffer_length,
	const cufftComplex* gpu_special_queue_end,
	const unsigned int start_index,
	const unsigned int max_index,
	const unsigned int frame_resolution,
	const unsigned int i_width,
	const unsigned int nsamples,
	const unsigned int n_i)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int size = frame_resolution * nsamples;

	while (index < frame_resolution)
	{
		cufftComplex M = make_cuComplex(0, 0);
		cufftComplex D = make_cuComplex(0, 0);
		int deplacement = (index  + (1 + i_width + ((1 + start_index) % max_index) *  frame_resolution) * (nsamples / 2)) % gpu_special_queue_buffer_length;
		cufftComplex b = gpu_special_queue[deplacement];

		for (int k = 0; k < nsamples; ++k)
		for (int j = 0; j < nsamples; ++j)
		for (int i = 0; i < nsamples; ++i)
		{
			deplacement = (index + i + (j * i_width) + (((k + start_index) % max_index) * frame_resolution)) % gpu_special_queue_buffer_length; // while x while y, on peut virer le modulo
			cufftComplex a = gpu_special_queue[deplacement];
			M.x += a.x;
			M.y += a.y;
			D.x += std::sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2)); // |a - b|
		}
		M.x += (n_i * b.x);
		M.y += (n_i * b.y);
		M.x /= D.x;
		M.y /= D.x;
		M.x = pow(M.x, 2);
		M.y = pow(M.y, 2);
	/*	float tmp = pow(M.x, 2) + pow(M.y, 2);
		M.x = (M.x * D.x) / tmp;
		M.y = (M.y * D.x) / tmp;
		tmp = M.x;
		M.x = pow(M.x, 2) - pow(M.y, 2);
		M.y = 2 * tmp * M.y;*/
		input[index] = M;
		index += blockDim.x * gridDim.x;
	}
}


void convolution_flowgraphy(
	cufftComplex* input,
	cufftComplex* gpu_special_queue,
	unsigned int &gpu_special_queue_start_index,
	const unsigned int gpu_special_queue_max_index,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
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
		sizeof(cufftComplex) * frame_resolution,
		cudaMemcpyDeviceToDevice);

	unsigned int n = pow(nframes, 3) - 3;
	unsigned int  gpu_special_queue_buffer_length = gpu_special_queue_max_index * frame_resolution;
	cufftComplex* gpu_special_queue_end = gpu_special_queue + gpu_special_queue_buffer_length;

	kernel_flowgraphy << <blocks, threads, 0, stream >> >(
		input,
		gpu_special_queue,
		gpu_special_queue_buffer_length,
		gpu_special_queue_end,
		gpu_special_queue_start_index,
		gpu_special_queue_max_index,
		frame_resolution,
		frame_width,
		nframes,
		n
		);

	cudaStreamSynchronize(stream);
}