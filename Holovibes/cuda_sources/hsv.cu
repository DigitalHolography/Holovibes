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

# include <stdio.h>
# include <fstream>
# include <nppdefs.h>
# include <nppcore.h>
# include <nppi.h>
# include <npps.h>
# include <nppversion.h>
# include <npp.h>
# include "min_max.cuh"
# include "tools_conversion.cuh"
# include "unique_ptr.hh"
# include "tools_compute.cuh"
# include "percentile.cuh"

# define SAMPLING_FREQUENCY  1




/*
* \brief Convert an array of HSV normalized float to an array of RGB normalized float
* i.e.:
* with "[  ]" a pixel:
* [HSV][HSV][HSV][HSV] -> [RGB][RGB][RGB][RGB]
* NVdia function
*/

__global__
void kernel_normalized_convert_hsv_to_rgb(const Npp32f* src, Npp32f* dst, size_t frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		Npp32f nNormalizedH = (Npp32f)src[id * 3];
		Npp32f nNormalizedS = (Npp32f)src[id * 3 + 1];
		Npp32f nNormalizedV = (Npp32f)src[id * 3 + 2];
		Npp32f nR;
		Npp32f nG;
		Npp32f nB;
		if (nNormalizedS == 0.0F)
		{
			nR = nG = nB = nNormalizedV;

		}
		else
		{
			if (nNormalizedH == 1.0F)
				nNormalizedH = 0.0F;
			else
				nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F

		}
		Npp32f nI = floorf(nNormalizedH);
		Npp32f nF = nNormalizedH - nI;
		Npp32f nM = nNormalizedV * (1.0F - nNormalizedS);
		Npp32f nN = nNormalizedV * (1.0F - nNormalizedS * nF);
		Npp32f nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
		if (nI == 0.0F)
		{
			nR = nNormalizedV; nG = nK; nB = nM;
		}
		else if (nI == 1.0F)
		{
			nR = nN; nG = nNormalizedV; nB = nM;
		}
		else if (nI == 2.0F)
		{
			nR = nM; nG = nNormalizedV; nB = nK;
		}
		else if (nI == 3.0F)
		{
			nR = nM; nG = nN; nB = nNormalizedV;
		}
		else if (nI == 4.0F)
		{
			nR = nK; nG = nM; nB = nNormalizedV;
		}
		else if (nI == 5.0F)
		{
			nR = nNormalizedV; nG = nM; nB = nN;
		}
		dst[id * 3] = nR;
		dst[id * 3 + 1] = nG;
		dst[id * 3 + 2] = nB;
	}
}

/*
** \brief Compute H component of hsv.
** h, s and v are separated
** we didn't make more loop to avoid jumps
*/
__global__
void kernel_compute_and_fill_hsv(const cuComplex* input, float* output, const size_t frame_res,
	const size_t min_index, const size_t max_index,
	const size_t diff_index, const size_t omega_size,
	const float* omega_arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		const size_t index_H = id * 3;
		const size_t index_S = id * 3 + 1;
		const size_t index_V = id * 3 + 2;
		output[index_H] = 0;
		output[index_S] = 0.95f;
		output[index_V] = 0;

		float min = FLT_MAX;
		for (size_t i = min_index; i <= max_index; ++i)
		{
			float input_elm = fabsf(input[i * frame_res + id].x);

			min = fminf(min, input_elm);
			output[index_H] += input_elm * omega_arr[i];
			output[index_V] += input_elm;
		}
		float summ_p = output[index_V];

		output[index_H] -= (max_index - min_index + 1) * min;
		output[index_H] /= summ_p;
	}
}


__global__
void threshold_top_bottom(float* output, const float tmin, const float tmax, const uint frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id] = fminf(output[id], tmax);
		output[id] = fmaxf(output[id], tmin);
	}
}

__global__
void from_distinct_components_to_interweaved_components(const Npp32f* src, Npp32f* dst, size_t frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		dst[id * 3] = src[id];
		dst[id * 3 + 1] = src[id + frame_res];
		dst[id * 3 + 2] = src[id + frame_res * 2];
	}
}

__global__
void from_interweaved_components_to_distinct_components(const Npp32f* src, Npp32f* dst, size_t frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		dst[id] = src[id * 3];
		dst[id + frame_res] = src[id * 3 + 1];
		dst[id + frame_res * 2] = src[id * 3 + 2];
	}
}


__global__
void kernel_fill_square_frequency_axis(const size_t length, float* arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < length)
	{
		arr[length + id] = arr[id] * arr[id];
	}
}

__global__
void kernel_fill_part_frequency_axis(const size_t min, const size_t max,
	const double step, const double origin, float* arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (min + id < max)
	{
		arr[min + id] = origin + id * step;
	}
}


void hsv(const cuComplex *d_input,
	float *d_output,
	const uint width,
	const uint height,
	uint index_min,
	uint index_max,
	uint nb_img,
	const float h,
	const float s,
	const float v,
	const float minH,
	const float maxH)
{
	const uint frame_res = height * width;

	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_res, threads);

	static float* omega_arr_data = nullptr;
	static size_t omega_arr_size = 0;

	static float* tmp_hsv_arr = nullptr;
	static size_t tmp_hsv_size = 0;

	index_max = std::min(index_max, nb_img - 1);
	index_min = std::min(index_min, index_max);

	if (tmp_hsv_size != frame_res)
	{
		tmp_hsv_size = frame_res;
		if (tmp_hsv_arr)
		{
			cudaFree(tmp_hsv_arr);
			cudaCheckError();
		}
		cudaMalloc(&tmp_hsv_arr, sizeof(float) * frame_res * 3 * 2); // HSV array * 2 , second part is for parallel reduction
	}

	if (omega_arr_size != nb_img)
	{
		omega_arr_size = nb_img;
		if (omega_arr_data)
		{
			cudaFree(omega_arr_data);
			cudaCheckError();
		}

		cudaMalloc(&omega_arr_data, sizeof(float) * nb_img * 2); // w1[] && w2[]
		cudaCheckError();

		double step = SAMPLING_FREQUENCY / (double)nb_img;
		size_t after_mid_index = nb_img / (double)2.0 + (double)1.0;
		kernel_fill_part_frequency_axis << <blocks, threads, 0, 0 >> > (0, after_mid_index, step, 0, omega_arr_data);
		double negative_origin = -SAMPLING_FREQUENCY / (double)2.0;
		if (nb_img % 2)
			negative_origin += step / (double)2.0;
		else
			negative_origin += step;
		kernel_fill_part_frequency_axis << <blocks, threads, 0, 0 >> > (after_mid_index, nb_img, step,
			negative_origin, omega_arr_data);
		kernel_fill_square_frequency_axis << <blocks, threads, 0, 0 >> > (nb_img, omega_arr_data);
		cudaStreamSynchronize(0);
		cudaCheckError();
	}

	kernel_compute_and_fill_hsv << <blocks, threads, 0, 0 >> > (d_input, d_output, frame_res,
		index_min, index_max, index_max - index_min + 1, omega_arr_size, omega_arr_data);
	cudaCheckError();




	//------------------------------------------------------------//

	from_interweaved_components_to_distinct_components << <blocks, threads, 0, 0 >> > (d_output, tmp_hsv_arr, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();
	//float minn, maxx;
	/*get_minimum_maximum_in_image(tmp_hsv_arr, frame_res, &minn, &maxx);
	std::cout << "part 1 min is : " << minn << "max is : " << maxx << std::endl;*/
	//	threshold_top_bottom << <blocks, threads, 0, 0 >> >(tmp_hsv_arr, 200000, maxx, frame_res);
	//get_minimum_maximum_in_image(tmp_hsv_arr, frame_res, &minn, &maxx);
	//gpu_multiply_const(tmp_hsv_arr, frame_res, 1 / maxx);

	//	normalize_frame(tmp_hsv_arr, frame_res); // h
	normalize_frame(tmp_hsv_arr, frame_res); // h
	cudaCheckError();

	float percent_out[2];
	const float percent_in_h[2] = 
	{
		0.2f, 99.8f
	};

	percentile_float(tmp_hsv_arr, frame_res, percent_in_h, percent_out, 2);
	float min_index_percentile = percent_out[0];
	float max_index_percentile = percent_out[1];

	threshold_top_bottom << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, min_index_percentile, max_index_percentile, frame_res);
	gpu_multiply_const(tmp_hsv_arr, frame_res, -1); // h
	normalize_frame(tmp_hsv_arr, frame_res); // h

	threshold_top_bottom << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, minH, maxH, frame_res);
	cudaCheckError();


	normalize_frame(tmp_hsv_arr, frame_res); // h
	cudaCheckError();

	gpu_multiply_const(tmp_hsv_arr, frame_res, 0.66f);

	//normalize_frame(tmp_hsv_arr + frame_res, frame_res); // s
	gpu_multiply_const(tmp_hsv_arr + frame_res, frame_res, s);

	const float percent_in_v[2] =
	{
		0.5f, 99.5f
	};
	percentile_float(tmp_hsv_arr + frame_res * 2, frame_res, percent_in_v, percent_out, 2);
	min_index_percentile = percent_out[0];
	max_index_percentile = percent_out[1];

	threshold_top_bottom << <blocks, threads, 0, 0 >> > (tmp_hsv_arr + frame_res * 2, min_index_percentile, max_index_percentile, frame_res);


	normalize_frame(tmp_hsv_arr + frame_res * 2, frame_res); // v
	gpu_multiply_const(tmp_hsv_arr + frame_res * 2, frame_res, v); 
	cudaCheckError();

	from_distinct_components_to_interweaved_components << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, d_output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();

	kernel_normalized_convert_hsv_to_rgb << <blocks, threads, 0, 0 >> > (d_output, d_output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();
	//-------------------------------------------------------------------------------//
	gpu_multiply_const(d_output, frame_res * 3, 65536);
	cudaStreamSynchronize(0);
	cudaCheckError();

}
