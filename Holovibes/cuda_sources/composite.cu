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



__global__
static void kernel_composite(cuComplex			*input,
							float				*output,
							const uint			frame_res,
							ushort				pmin_r,
							ushort				pmax_r,
							float				weight_r,
							ushort				pmin_g,
							ushort				pmax_g,
							float				weight_g,
							ushort				pmin_b,
							ushort				pmax_b,
							float				weight_b)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	ushort pmin[] = { pmin_r, pmin_g, pmin_b };
	ushort pmax[] = { pmax_r, pmax_g, pmax_b };
	float weight[] = { weight_r, weight_g, weight_b };
	if (id < frame_res)
	{
		for (int i = 0; i < 3; i++)
		{
			float res = 0;
			for (ushort p = pmin[i]; p <= pmax[i]; p++)
			{
				cuComplex *current_pframe = input + (frame_res * p);
				res += hypotf(current_pframe[id].x, current_pframe[id].y);
			}
			output[id * 3 + i] = res * weight[i] / (pmax[i] - pmin[i] + 1);
		}
	}
}

// ! Splits the image by nb_lines blocks and sums them
__global__
static void kernel_sum_one_line(float			*input,
							const uint			frame_res,
							const uchar			pixel_depth,
							const uint			nb_lines,
							const uint			line_size,
							float				*sums_per_line)
{
	const uint	id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < pixel_depth * nb_lines)
	{
		uchar offset = id % pixel_depth;
		ushort line = id / pixel_depth;
		uint index_begin = line_size * line;
		uint index_end = line_size * (line + 1);
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
		input[id] /= averages[id % 3];
}

void composite(cuComplex	*input,
			float			*output,
			const uint		frame_res,
			bool			normalize,
			ushort			pmin_r,
			ushort			pmax_r,
			float			weight_r,
			ushort			pmin_g,
			ushort			pmax_g,
			float			weight_g,
			ushort			pmin_b,
			ushort			pmax_b,
			float			weight_b)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_res, threads);

	kernel_composite << <blocks, threads, 0, 0 >> > (input,
		output,
		frame_res,
		pmin_r,
		pmax_r,
		normalize ? 1 : weight_r,
		pmin_g,
		pmax_g,
		normalize ? 1 : weight_g,
		pmin_b,
		pmax_b,
		normalize ? 1 : weight_b);
	cudaCheckError();
	if (normalize)
	{
		const ushort line_size = 1024;
		const ushort lines = frame_res / line_size + 1;
		float *averages = nullptr;
		float *sums_per_line = nullptr;
		const uchar pixel_depth = 3;
		cudaMalloc(&averages, sizeof(float) * pixel_depth);
		cudaMalloc(&sums_per_line, sizeof(float) * lines * pixel_depth);
		blocks = map_blocks_to_problem(lines * pixel_depth, threads);
		kernel_sum_one_line << <blocks, threads, 0, 0 >> > (output,
			frame_res,
			pixel_depth,
			lines,
			line_size,
			sums_per_line);
	cudaCheckError();
		blocks = map_blocks_to_problem(pixel_depth, threads);
		kernel_average_float_array << <blocks, threads, 0, 0 >> > (output,
			lines,
			frame_res,
			pixel_depth,
			averages);
	cudaCheckError();
		blocks = map_blocks_to_problem(frame_res * pixel_depth, threads);
		kernel_divide_by_weight << <1, 1, 0, 0 >> > (averages, weight_r, weight_g, weight_b);
	cudaCheckError();
		kernel_normalize_array << <blocks, threads, 0, 0 >> > (output,
			frame_res,
			pixel_depth,
			averages);
	cudaCheckError();
		cudaFree(averages);
		cudaFree(sums_per_line);
	}
}
