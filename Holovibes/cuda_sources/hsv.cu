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

# include "hsv.cuh"
# include <stdio.h>
# define SAMPLING_FREQUENCY  1



/*
* \brief Convert an array of HSV normalized float to an array of RGB normalized float
* i.e.: 
* with "[  ]" a pixel: 
* [HSV][HSV][HSV][HSV] -> [RGB][RGB][RGB][RGB]
* This should be cache compliant

* This function has been taken from NVidia website and slightly modified
*/

__global__
void kernel_normalized_convert_hsv_to_rgb(const Npp32f* src, Npp32f* dst, size_t frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		const uint idd = id * 3;

		Npp32f nNormalizedH = src[idd] > 1 ? 1 : src[idd];
		Npp32f nNormalizedS = src[idd + 1] > 1 ? 1 : src[idd + 1];
		Npp32f nNormalizedV = src[idd + 2] > 1 ? 1 : src[idd + 2];

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

		dst[idd] = nR;
		dst[idd + 1] = nG;
		dst[idd + 2] = nB;
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
		output[index_S] = 0;
		output[index_V] = 0;

		for (size_t i = min_index; i <= max_index; ++i)
		{
			float input_elm = fabsf(input[i * frame_res + id].x);
			output[index_H] += input_elm * omega_arr[i];
			output[index_S] += input_elm * omega_arr[omega_size + i];
			output[index_V] += input_elm;
		}
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



void hsv(const cuComplex *input,
	float *output,
	const uint frame_res,
	uint index_min,
	uint index_max,
	uint nb_img,
	const float h,
	const float s,
	const float v,
	const float minH,
	const float maxH)
{
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
		kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(0, after_mid_index, step, 0, omega_arr_data);
		double negative_origin = -SAMPLING_FREQUENCY / (double)2.0;
		if(nb_img % 2)
			negative_origin += step / (double)2.0;
		else 
			negative_origin += step;
		kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(after_mid_index, nb_img, step,
																  negative_origin, omega_arr_data);
		kernel_fill_square_frequency_axis <<<blocks, threads, 0, 0 >>> (nb_img, omega_arr_data);
		cudaStreamSynchronize(0);
		cudaCheckError();
	}
	
	kernel_compute_and_fill_hsv <<<blocks, threads, 0, 0 >> > (input, output, frame_res,
		 index_min, index_max,index_max - index_min + 1, omega_arr_size, omega_arr_data);
	cudaStreamSynchronize(0);
	cudaCheckError();

	from_interweaved_components_to_distinct_components << <blocks, threads, 0, 0 >> > (output, tmp_hsv_arr, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();

	normalize_frame(tmp_hsv_arr, frame_res); // h
	cudaCheckError();

	threshold_top_bottom << <blocks, threads, 0, 0 >> >(tmp_hsv_arr, minH, maxH, frame_res);
	cudaCheckError();

	normalize_frame(tmp_hsv_arr, frame_res); // h
	gpu_multiply_const(tmp_hsv_arr, frame_res, h);

	normalize_frame(tmp_hsv_arr + frame_res, frame_res); // s
	gpu_multiply_const(tmp_hsv_arr + frame_res, frame_res, s);

	normalize_frame(tmp_hsv_arr + frame_res * 2, frame_res); // v
	gpu_multiply_const(tmp_hsv_arr + frame_res * 2, frame_res, v);

	cudaCheckError();

	from_distinct_components_to_interweaved_components << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();


	//kernel_normalized_convert_hsv_to_rgb << <blocks, threads, 0, 0 >> > (output, output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();
	
	gpu_multiply_const(output, frame_res * 3, 65025);
	cudaStreamSynchronize(0);
	cudaCheckError();
	
}
