#include <device_launch_parameters.h>

#include "flowgraphy.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

__global__ void kernel_multiply_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_resolution,
	const unsigned int i_width,
	const unsigned int nsamples)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int size = frame_resolution * nsamples;
	while (index < frame_resolution)
	{
		cufftComplex M = make_cuComplex(0, 0);
		cufftComplex D = make_cuComplex(0, 0);
		cufftComplex b = tmp_input[(index + 1 + i_width + frame_resolution)];
		for (int k = 0; k < 3; ++k)
		for (int j = 0; j < 3; ++j)
		for (int i = 0; i < 3; ++i)
		{
			cufftComplex a = tmp_input[(index + i + (j * i_width) + (k * frame_resolution)) % size];
			M.x += a.x;
			M.y += a.y;
			D.x += std::sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
			D.y = 0;
		}
		M.x += (24 * b.x);
		M.y += (24 * b.y);
		M.x /= D.x;
		M.y /= D.x;
		M.x = pow(M.x, 2);
		M.y = pow(M.y, 2);
		input[index] = M;
		index += blockDim.x * gridDim.x;
	}
}

void convolution_flowgraphy(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
	cudaStream_t stream)
{
	// const unsigned int n_frame_resolution = frame_resolution * nframes;

	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);

	cudaMemcpy(tmp_input, input, sizeof(cufftComplex)* frame_resolution * nframes, cudaMemcpyDeviceToDevice);

	kernel_multiply_kernel << <blocks, threads, 0, stream >> >(
		input,
		tmp_input,
		frame_resolution,
		frame_width,
		nframes
		);

	cudaStreamSynchronize(stream);
}