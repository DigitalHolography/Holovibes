#include <device_launch_parameters.h>
#include <cmath>

#include "average.cuh"
#include "geometry.hh"
#include "tools.hh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "hardware_limits.hh"

#define THREADS 256

/*! \brief  Sume 2 zone of input image
*
* \param input The image from where zones should be summed.
* \param width The width of the input image.
* \param height The height of the input image.
*
*/
static __global__ void kernel_zone_sum(
	float* input,
	const unsigned int width,
	float* output,
	const unsigned int zone_start_x,
	const unsigned int zone_start_y,
	const unsigned int zone_width,
	const unsigned int zone_height)
{
	const unsigned int size = zone_width * zone_height;
	unsigned int tid = threadIdx.x;
	unsigned int index = blockIdx.x * blockDim.x + tid;
	extern __shared__ float  sdata[];

	// INIT
	sdata[tid] = 0.0f;

	// SUM input in sdata
	while (index < size)
	{
		int x = index % zone_width + zone_start_x;
		int y = index / zone_width + zone_start_y;
		int index2 = y * width + x;

		sdata[tid] += input[index2];
		index += blockDim.x * gridDim.x;
	}

	// Sum sdata in sdata[0]
	__syncthreads();
	for (unsigned int s = blockDim.x >> 1; s > 32; s >>= 1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	// Return result
	__syncthreads();
	if (tid == 0)
		*output = sdata[0];
}

std::tuple<float, float, float, float> make_average_plot(
	float *input,
	const unsigned int width,
	const unsigned int height,
	const holovibes::Rectangle& signal,
	const holovibes::Rectangle& noise,
	cudaStream_t stream)
{
	//const unsigned int size = width * height;
	unsigned int threads = THREADS;
	//unsigned int blocks = map_blocks_to_problem(size, threads);

	float* gpu_s;
	float* gpu_n;

	cudaMalloc(&gpu_s, sizeof(float));
	cudaMalloc(&gpu_n, sizeof(float));

	cudaMemsetAsync(gpu_s, 0, sizeof(float), stream);
	cudaMemsetAsync(gpu_n, 0, sizeof(float), stream);

	unsigned int signal_width = abs(signal.top_right.x - signal.top_left.x);
	unsigned int signal_height = abs(signal.top_left.y - signal.bottom_left.y);
	unsigned int noise_width = abs(noise.top_right.x - noise.top_left.x);
	unsigned int noise_height = abs(noise.top_left.y - noise.bottom_left.y);

	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_n,
		noise.top_left.x, noise.top_left.y, noise_width, noise_height);
	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_s,
		signal.top_left.x, signal.top_left.y, signal_width, signal_height);

	float cpu_s;
	float cpu_n;

	cudaMemcpyAsync(&cpu_s, gpu_s, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&cpu_n, gpu_n, sizeof(float), cudaMemcpyDeviceToHost, stream);

	cpu_s /= float(signal_width * signal_height);
	cpu_n /= float(noise_width * noise_height);

	float moy = cpu_s / cpu_n;

	cudaFree(gpu_n);
	cudaFree(gpu_s);

	return std::tuple < float, float, float, float > { cpu_s, cpu_n, moy, 10 * log10f(moy)};
}

std::tuple<float, float, float, float> make_average_stft_plot(
	cufftComplex*          cbuf,
	float*                 fbuf,
	cufftComplex*          stft_buffer,
	const unsigned int     width,
	const unsigned int     height,
	const unsigned int     width_roi,
	const unsigned int     height_roi,
	holovibes::Rectangle&  signal_zone,
	holovibes::Rectangle&  noise_zone,
	const unsigned int     pindex,
	const unsigned int     nsamples,
	cudaStream_t stream)
{
	std::tuple<float, float, float, float> res;

	const unsigned int size = width * height;
	//unsigned int threads = 128;
	//unsigned int blocks = map_blocks_to_problem(size, threads);

	// Reconstruct Roi
	/*kernel_reconstruct_roi << <blocks, threads, 0, stream >> >(
	  stft_buffer,
	  cbuf,
	  width_roi,
	  height_roi,
	  width,
	  width,
	  height,
	  pindex,
	  nsamples);*/

	complex_to_modulus(cbuf, fbuf, size, stream);

	return make_average_plot(fbuf, width, height, signal_zone, noise_zone, stream);
}