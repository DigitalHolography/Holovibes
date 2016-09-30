#include <device_launch_parameters.h>

#include "convolution.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_multiply_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_width,
	const unsigned int i_width,
	const cufftComplex* kernel,
	const unsigned int k_width,
	const unsigned int k_height,
	const unsigned int nsamples)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int n, m, z;
	unsigned int size = frame_width * nsamples;
	while (index < size)
	{
		cufftComplex sum = make_cuComplex(0, 0);
		for (z = 0; z < nsamples; ++z)
		for (m = 0; m < k_width; ++m)
		for (n = 0; n < k_height; ++n) {
			cufftComplex a = tmp_input[(index + m + n * i_width + z * frame_width) % size];
			cufftComplex b = kernel[m + n * k_width + (z * k_width * k_height)];

			sum.x += a.x * b.x - a.y * b.y;
			sum.y += a.y * b.x + a.x * b.y;
			sum.x /= nsamples * k_width * k_height;
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
	const unsigned int k_height,
	const unsigned int k_z,
	cudaStream_t stream)
{
	// const unsigned int n_frame_resolution = frame_resolution * nframes;

	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);
	
	cudaMemcpy(tmp_input, input, sizeof(cufftComplex) * frame_resolution * k_z, cudaMemcpyDeviceToDevice);

	kernel_multiply_kernel<<<blocks, threads, 0, stream>>>(
		input,
		tmp_input,
		frame_resolution,
		frame_width,
		kernel,
		k_width,
		k_height,
		k_z
		);

	cudaStreamSynchronize(stream);
}