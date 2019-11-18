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

#include "average.cuh"
#include "tools_conversion.cuh"
#include "units/rect.hh"
#include "unique_ptr.hh"
#include "tools.hh"

using holovibes::units::RectFd;
using holovibes::Tuple4f;

/*static __global__ 
void kernel_zone_sum(const float	*input,
					const uint		width,
					float			*output,
					const uint		zTopLeft_x,
					const uint		zTopLeft_y,
					const uint		zone_width,
					const uint		zone_height)
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
}*/


static __global__
void kernel_zone_sum(const float	*input,
	const uint		width,
	double			*output,
	const uint		zTopLeft_x,
	const uint		zTopLeft_y,
	const uint		zone_width,
	const uint		zone_height)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < zone_height)
	{
		double sum = 0.;
		const float* line = input + width * (index + zTopLeft_y);
		for (uint i = 0; i < zone_width; i++)
			sum += line[i + zTopLeft_x];

		output[index] = sum / zone_width;
	}
}

static __global__
void kernel_compute_average_line(const double *input, const uint size, double *output)
{
	*output = 0;
	for (uint i = 0; i < size; i++)
		*output += input[i];
	*output /= size;
}

static
void zone_sum(const float *input,
			  const uint width,
			  double *output,
			  const RectFd& zone,
			  cudaStream_t stream = 0)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(zone.height(), threads);

	holovibes::cuda_tools::UniquePtr<double> output_buf;
	output_buf.resize(zone.height());

	// Average each line
	kernel_zone_sum << <blocks, threads, 0, stream >> > (input, width, output_buf, zone.topLeft().x(), zone.topLeft().y(), zone.width(), zone.height());
	cudaCheckError();
	// Average of lines
	kernel_compute_average_line << < 1, 1, 0, stream>> > (output_buf, 1, output);
	cudaStreamSynchronize(stream);
	cudaCheckError();
}

Tuple4f make_average_plot(float				*input,
						const uint			width,
						const uint			height,
						const RectFd&		signal,
						const RectFd&		noise,
						cudaStream_t		stream)
{
	//const uint size = width * height;
	//const uint threads = THREADS_256;
	//uint blocks = map_blocks_to_problem(size, threads);

	holovibes::cuda_tools::UniquePtr<double> gpu_s;
	holovibes::cuda_tools::UniquePtr<double> gpu_n;

	if (!gpu_s.resize(1))
		return std::make_tuple(0.f, 0.f, 0.f, 0.f);
	if (!gpu_n.resize(1))
		return std::make_tuple(0.f, 0.f, 0.f, 0.f);

	zone_sum(input, width, gpu_s, signal);
	zone_sum(input, width, gpu_n, noise);

	/*cudaMemsetAsync(gpu_s, 0, sizeof(float), stream);
	cudaMemsetAsync(gpu_n, 0, sizeof(float), stream);

	const uint signal_width = signal.width();
	const uint signal_height = signal.height();
	const uint noise_width = noise.width();
	const uint noise_height = noise.height();

	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_s,
		signal.topLeft().x(), signal.topLeft().y(), signal_width, signal_height);
	cudaCheckError();
	kernel_zone_sum << <1, threads, threads * sizeof(float), stream >> >(input, width, gpu_n,
		noise.topLeft().x(), noise.topLeft().y(), noise_width, noise_height);
	cudaCheckError();

	float cpu_s;
	float cpu_n;

	cudaMemcpyAsync(&cpu_s, gpu_s, sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&cpu_n, gpu_n, sizeof(float), cudaMemcpyDeviceToHost, stream);

	cudaFree(gpu_n);
	cudaFree(gpu_s);

	cpu_s /= static_cast<float>(signal_width * signal_height);
	cpu_n /= static_cast<float>(noise_width * noise_height);*/

	double cpu_s;
	double cpu_n;

	cudaMemcpy(&cpu_s, gpu_s.get(), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&cpu_n, gpu_n.get(), sizeof(double), cudaMemcpyDeviceToHost);

	float moy = cpu_s / cpu_n;

	return Tuple4f{ cpu_s, cpu_n, moy, 10 * log10f(moy)};
}

Tuple4f make_average_stft_plot(cuComplex	*cbuf,
							float			*fbuf,
							cuComplex		*stft_buffer,
							const uint		width,
							const uint		height,
							const uint		width_roi,
							const uint		height_roi,
							const holovibes::units::RectFd&	signal_zone,
							const holovibes::units::RectFd&	noise_zone,
							const uint		pindex,
							const uint		nSize,
							cudaStream_t	stream)
{
	const uint	size = width * height;

	complex_to_modulus(cbuf, fbuf, stft_buffer, size, pindex, pindex, stream);

	return make_average_plot(fbuf, width, height, signal_zone, noise_zone, stream);
}



__global__
void kernel_average_lines(float*	input,
						float*	output,
						uint	width,
						uint	height)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < height)
	{
		float sum = 0.f;
		float* line = input + width * index;
		for (uint i = 0; i < width; i++)
			sum += line[i];
		output[index] = sum / width;
	}
}


void average_lines(float*	input,
					float*	output,
					uint	width,
					uint	height)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(height, threads);
	kernel_average_lines << <blocks, threads, 0, 0 >> > (input, output, width, height);
	cudaCheckError();
}
