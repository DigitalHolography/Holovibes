#include <device_launch_parameters.h>

#include "convolution.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_multiply_kernel(	complex		*input,
										complex		*gpu_special_queue,
										const uint	gpu_special_queue_buffer_length,
										const uint	frame_resolution,
										const uint	i_width,
										const float	*kernel,
										const uint	k_width,
										const uint	k_height,
										const uint	nsamples,
										const uint	start_index,
										const uint	max_index)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint n, m, z;
	//uint size = frame_resolution * nsamples;
	uint k_size = k_width * k_height;
	while (index < frame_resolution)
	{
		complex sum = make_cuComplex(0, 0);

		for (z = 0; z < nsamples; ++z)
		for (m = 0; m < k_width; ++m)
		for (n = 0; n < k_height; ++n) {
			complex a = gpu_special_queue[(index + m + n * i_width + (((z + start_index) % max_index) * frame_resolution)) % gpu_special_queue_buffer_length];
			float b = kernel[m + n * k_width + (z * k_size)];
			sum.x += a.x * b;
			sum.y += a.y * b;
		}
		const uint n_k_size = nsamples * k_size;
		sum.x /= n_k_size;
		sum.y /= n_k_size;
		input[index] = sum;
		index += blockDim.x * gridDim.x;
	}
}

void convolution_kernel(complex			*input,
						complex			*gpu_special_queue,
						const uint		frame_resolution,
						const uint		frame_width,
						const float		*kernel,
						const uint		k_width,
						const uint		k_height,
						const uint		k_z,
						uint&			gpu_special_queue_start_index,
						const uint&		gpu_special_queue_max_index,
						cudaStream_t	stream)
{

	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);

	if (gpu_special_queue_start_index == 0)
		gpu_special_queue_start_index = gpu_special_queue_max_index - 1;
	else
		--gpu_special_queue_start_index;
	cudaMemcpy(
		gpu_special_queue + frame_resolution * gpu_special_queue_start_index,
		input,
		sizeof(complex) * frame_resolution,
		cudaMemcpyDeviceToDevice);
	
	uint gpu_special_queue_buffer_length = gpu_special_queue_max_index * frame_resolution;

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