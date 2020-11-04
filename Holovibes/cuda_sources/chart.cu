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

#include "chart.cuh"
#include "tools_conversion.cuh"
#include "units/rect.hh"
#include "unique_ptr.hh"
#include "tools.hh"
#include "cuda_memory.cuh"

using holovibes::units::RectFd;
using holovibes::Tuple4f;

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

Tuple4f make_chart_plot(float				*input,
						const uint			width,
						const uint			height,
						const RectFd&		signal,
						const RectFd&		noise,
						cudaStream_t		stream)
{
	holovibes::cuda_tools::UniquePtr<double> gpu_s;
	holovibes::cuda_tools::UniquePtr<double> gpu_n;

	if (!gpu_s.resize(1))
		return std::make_tuple(0.f, 0.f, 0.f, 0.f);
	if (!gpu_n.resize(1))
		return std::make_tuple(0.f, 0.f, 0.f, 0.f);

	zone_sum(input, width, gpu_s, signal);
	zone_sum(input, width, gpu_n, noise);

	double cpu_s;
	double cpu_n;

	cudaXMemcpy(&cpu_s, gpu_s.get(), sizeof(double), cudaMemcpyDeviceToHost);
	cudaXMemcpy(&cpu_n, gpu_n.get(), sizeof(double), cudaMemcpyDeviceToHost);

	float moy = cpu_s / cpu_n;

	return Tuple4f{ cpu_s, cpu_n, moy, 10 * log10f(moy)};
}
