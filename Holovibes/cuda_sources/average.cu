/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include <qpoint.h>

#include "average.cuh"
#include "tools.hh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "hardware_limits.hh"

/*! \brief  Sume 2 zone of input image
*
* \param input The image from where zones should be summed.
* \param width The width of the input image.
* \param height The height of the input image.
*
*/
static __global__ void kernel_zone_sum(	const float	*input,
										const uint	width,
										float		*output,
										const uint	zTopLeft_x,
										const uint	zTopLeft_y,
										const uint	zone_width,
										const uint	zone_height)
{
	const uint				size = zone_width * zone_height;
	const uint				tid = threadIdx.x;
	const uint				index = blockIdx.x * blockDim.x + tid;
	extern __shared__ float	sdata[];

	// INIT
	sdata[tid] = 0.0f;

	// SUM input in sdata
	if (index < size)
	{
		int x = index % zone_width + zTopLeft_x;
		int y = index / zone_width + zTopLeft_y;
		int index2 = y * width + x;

		sdata[tid] += input[index2];
	}

	// Sum sdata in sdata[0]
	__syncthreads();
	for (uint s = blockDim.x >> 1; s > 32; s >>= 1)
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

std::tuple<float, float, float, float> make_average_plot(	float						*input,
															const uint					width,
															const uint					height,
															const gui::Rectangle&	signal,
															const gui::Rectangle&	noise,
															cudaStream_t				stream)
{
	//const uint size = width * height;
	const uint threads = THREADS_256;
	//uint blocks = map_blocks_to_problem(size, threads);

	float *gpu_s;
	float *gpu_n;

	cudaMalloc(&gpu_s, sizeof(float));
	cudaMalloc(&gpu_n, sizeof(float));

	cudaMemsetAsync(gpu_s, 0, sizeof(float), stream);
	cudaMemsetAsync(gpu_n, 0, sizeof(float), stream);

	const uint signal_width = signal.width();
	const uint signal_height = signal.height();
	const uint noise_width = noise.width();
	const uint noise_height = noise.height();

	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_n,
		noise.topLeft().x(), noise.topLeft().y(), noise_width, noise_height);
	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_s,
		signal.topLeft().x(), signal.topLeft().y(), signal_width, signal_height);

	float cpu_s;
	float cpu_n;

	cudaMemcpyAsync(&cpu_s, gpu_s, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&cpu_n, gpu_n, sizeof(float), cudaMemcpyDeviceToHost, stream);

	cpu_s /= static_cast<float>(signal_width * signal_height);
	cpu_n /= static_cast<float>(noise_width * noise_height);

	float moy = cpu_s / cpu_n;

	cudaFree(gpu_n);
	cudaFree(gpu_s);

	return std::tuple < float, float, float, float > { cpu_s, cpu_n, moy, 10 * log10f(moy)};
}

std::tuple<float, float, float, float> make_average_stft_plot(	complex					*cbuf,
																float					*fbuf,
																complex					*stft_buffer,
																const uint				width,
																const uint				height,
																const uint				width_roi,
																const uint				height_roi,
																gui::Rectangle&	signal_zone,
																gui::Rectangle&	noise_zone,
																const uint				pindex,
																const uint				nsamples,
																cudaStream_t			stream)
{
	std::tuple<float, float, float, float> res;

	const uint size = width * height;
	//uint threads = 128;
	//uint blocks = map_blocks_to_problem(size, threads);

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