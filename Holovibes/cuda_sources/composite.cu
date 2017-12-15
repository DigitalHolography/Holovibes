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

#include "tools_conversion.cuh"
#include "color.hh"

struct rect
{
	int x;
	int y;
	int w;
	int h;
};

struct comp
{
	ushort p_min;
	ushort p_max;
	float weight;
};


namespace
{
	void check_zone(rect& zone, const uint frame_res, const int line_size)
	{
		const int lines = line_size ? frame_res / line_size : 0;
		if (!zone.h || !zone.w || zone.x + zone.w > line_size || zone.y + zone.h > lines)
		{
			zone.x = 0;
			zone.y = 0;
			zone.w = line_size;
			zone.h = frame_res / line_size;
		}
	}
}
__global__
static void kernel_composite(cuComplex			*input,
							float				*output,
							const uint			frame_res,
							bool				normalize,
							size_t				red,
							size_t				blue)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		double res_components[3] = { 0 };
		ushort min = red < blue ? red : blue;
		ushort max = blue > red ? blue : red;
		ushort range = std::abs(static_cast<short>(blue - red)) + 1;
		for (ushort p = min; p <= max; p++)
		{
			double components_[3];
			double x = (p - min) / double(range);
			if (red > blue)
				x = 1 - x;
			if (x < 0.25) {
				components_[0] = 1;
				components_[1] = x / 0.25;
				components_[2] = 0;
			}
			else if (x < 0.5) {
				components_[0] = 1 - (x - 0.25) / 0.25;
				components_[1] = 1;
				components_[2] = 0;
			}
			else if (x < 0.75) {
				components_[0] = 0;
				components_[1] = 1;
				components_[2] = (x - 0.5) / 0.25;
			}
			else {
				components_[0] = 0;
				components_[1] = 1 - (x - 0.75) / 0.25;
				components_[2] = 1;
			}
			cuComplex *current_pframe = input + (frame_res * p);
			float intensity = hypotf(current_pframe[id].x, current_pframe[id].y);
			for (int i = 0; i < 3; i++)
				res_components[i] += components_[i] * intensity;
		}
		for (int i = 0; i < 3; i++)
			output[id * 3 + i] = res_components[i] /** weight[i]*/ / double(range);
	}
}

// ! Splits the image by nb_lines blocks and sums them
__global__
static void kernel_sum_one_line(float			*input,
							const uint			frame_res,
							const uchar			pixel_depth,
							const uint			line_size,
							const rect			zone,
							float				*sums_per_line)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < pixel_depth * zone.h)
	{
		uchar offset = id % pixel_depth;
		ushort line = id / pixel_depth;
		line += zone.y;
		uint index_begin = line_size * line + zone.x;
		uint index_end = index_begin + zone.w;
		if (index_end > frame_res)
			index_end = frame_res;
		float sum = 0;
		while(index_begin < index_end)
			sum += input[pixel_depth * (index_begin++) + offset];
		sums_per_line[id] = sum;
	}
}

// ! sums an array of size floats and put the result divided by nb_elements in *output
__global__
static void kernel_average_float_array(float		*input,
								uint				size,
								uint				nb_elements,
								uint				offset_per_pixel,
								float				*output)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < offset_per_pixel)
	{
		input += id;
		float res = 0;
		while (size--)
		{
			res += *input;
			input += offset_per_pixel;
		}
		res /= static_cast<float>(nb_elements);
		output[id] = res;
	}
}

__global__
static void kernel_divide_by_weight(float		*input,
							float				weight_r,
							float				weight_g,
							float				weight_b)
{
	input[0] /= weight_r;
	input[1] /= weight_g;
	input[2] /= weight_b;
}
__global__
static void kernel_normalize_array(float			*input,
								uint				nb_pixels,
								uint				pixel_depth,
								float				*averages)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < pixel_depth * nb_pixels)
		input[id] /= averages[id % 3] / 1000;
	// The /1000 is used to have the result in [0;1000]
	// instead of [0;1] for a better contrast control
}

void composite(cuComplex	*input,
	float					*output,
	const uint				frame_res,
	const uint				real_line_size,
	bool					normalize,
	holovibes::units::RectFd	selection,
	const size_t red,
	const size_t blue)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_composite << <blocks, threads, 0, 0 >> > (input, output, frame_res, normalize, red, blue);
	cudaCheckError();
	cudaStreamSynchronize(0);
	if (normalize)
	{
		rect zone = { selection.x(), selection.y(), selection.unsigned_width(), selection.unsigned_height() };
		check_zone(zone, frame_res, real_line_size);
		const ushort line_size = zone.w;
		const ushort lines = zone.h;
		float *averages = nullptr;
		float *sums_per_line = nullptr;
		const uchar pixel_depth = 3;
		cudaMalloc(&averages, sizeof(float) * pixel_depth);
		cudaMalloc(&sums_per_line, sizeof(float) * lines * pixel_depth);


		blocks = map_blocks_to_problem(lines * pixel_depth, threads);
		kernel_sum_one_line << <blocks, threads, 0, 0 >> > (output,
			frame_res,
			pixel_depth,
			real_line_size,
			zone,
			sums_per_line);
		cudaCheckError();
		cudaStreamSynchronize(0);

		blocks = map_blocks_to_problem(pixel_depth, threads);
		kernel_average_float_array << <blocks, threads, 0, 0 >> > (sums_per_line,
			lines,
			lines * line_size,
			pixel_depth,
			averages);
		cudaCheckError();
		cudaStreamSynchronize(0);

		blocks = map_blocks_to_problem(frame_res * pixel_depth, threads);
		//kernel_divide_by_weight << <1, 1, 0, 0 >> > (averages, r.weight, g.weight, b.weight);
		cudaCheckError();
		cudaStreamSynchronize(0);
		kernel_normalize_array << <blocks, threads, 0, 0 >> > (output,
			frame_res,
			pixel_depth,
			averages);
		cudaCheckError();
		cudaStreamSynchronize(0);
		cudaFree(averages);
		cudaFree(sums_per_line);
	}
}
