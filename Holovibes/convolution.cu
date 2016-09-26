#include <device_launch_parameters.h>

#include "convolution.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_multiply_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int size,
	const unsigned int i_width,
	const cufftComplex* kernel,
	const unsigned int k_width,
	const unsigned int k_height)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int n, m;

	while (index < size)
	{
		cufftComplex sum;

		sum.x = 0;
		sum.y = 0;
		for (m = 0; m < k_width; ++m)
		for (n = 0; n < k_height; ++n) {
			cufftComplex a = tmp_input[(index + m + n * i_width) % size];
			cufftComplex b = kernel[m + n * k_width];

			sum.x += a.x * b.x - a.y * b.y;
			sum.y += a.y * b.x + a.x * b.y;
		}
		input[index] = sum;
		index += blockDim.x * gridDim.x;
	}
}

void convolution_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
	const cufftComplex* kernel,
	const unsigned int k_width,
	const unsigned int k_heigth,
	const unsigned int k_z,
	cudaStream_t stream)
{
	// const unsigned int n_frame_resolution = frame_resolution * nframes;

	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);
	
	cudaMemcpy(tmp_input, input, sizeof(cufftComplex) * frame_resolution, cudaMemcpyDeviceToDevice);

	kernel_multiply_kernel<<<blocks, threads, 0, stream>>>(
		input,
		tmp_input,
		frame_resolution,
		frame_width,
		kernel,
		k_width,
		k_heigth
		);

	cudaStreamSynchronize(stream);
}