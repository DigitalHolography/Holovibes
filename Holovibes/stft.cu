#include <device_launch_parameters.h>

#include "stft.cuh"
#include "hardware_limits.hh"
#include "geometry.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"
#include "geometry.hh"

void stft(
	cufftComplex*                   input,
	cufftComplex*                   gpu_queue,
	cufftComplex*                   stft_buf,
	const cufftHandle               plan1d,
	unsigned int                    stft_level,
	unsigned int                    p,
	unsigned int                    q,
	unsigned int                    frame_size,
	bool                            stft_activated,
	cudaStream_t                    stream)
{
	//unsigned int threads = 128;
	//unsigned int blocks = map_blocks_to_problem(frame_size, threads);

	// FFT 1D
	if (stft_activated)
		cufftExecC2C(plan1d, gpu_queue, stft_buf, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);

	cudaMemcpy(
		input,
		stft_buf + p * frame_size,
		sizeof(cufftComplex)* frame_size,
		cudaMemcpyDeviceToDevice);

	if (p != q)
	{
		cudaMemcpy(
			input + frame_size,
			stft_buf + q * frame_size,
			sizeof(cufftComplex)* frame_size,
			cudaMemcpyDeviceToDevice);

	}

}

__global__	void	stft_view_xz(	cufftComplex	*input,
									ushort			*output,
									uint			x0,
									uint			y0,
									uint			z0,
									uint			frame_size,
									uint			width,
									uint			height,
									uint			depth)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < frame_size)
	{
		uint index_x = id;
		uint index_z = id % depth;
//		cufftComplex pixel = input[(y0 * width) + (index_x / width) * frame_size + index_x % width];
		cufftComplex pixel = input[(y0 * width) + (index_x / width) * frame_size + index_x % width];
		float res = hypotf(pixel.x, pixel.y);
		output[id] = static_cast<ushort>(pixel.x);
	}
}

void	stft_view_begin(	cufftComplex	*input,
							ushort			*output,
							uint			x0,
							uint			y0,
							uint			z0,
							uint			frame_size,
							uint			width,
							uint			height,
							uint			depth)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_size, threads);

	stft_view_xz<<<blocks, threads, 0, 0 >>>(input, output, x0, y0, z0, frame_size, width, height, depth);
}
