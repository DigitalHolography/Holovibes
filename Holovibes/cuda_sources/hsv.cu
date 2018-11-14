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
# include <nppdefs.h>
# include <nppcore.h>
# include <nppi.h>
# include <npps.h>
# include <nppversion.h>
# include <npp.h>
# include <stdio.h>
# define SAMPLING_FREQUENCY  1


#define BELOW_ONE 0.99999f


/*
* \brief Convert an array of HSV normalized float to an array of RGB normalized float
* i.e.: 
* with "[  ]" a pixel: 
* [HSV][HSV][HSV][HSV] -> [RGB][RGB][RGB][RGB]
* This should be cache compliant
*/

__global__
void kernel_normalized_convert_hsv_to_rgb(const Npp32f* src, Npp32f* dst, size_t frame_res)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		const uint idd = id * 3;
		Npp32f nNormalizedH = fminf(src[idd], BELOW_ONE);
		Npp32f nNormalizedS = fminf(src[idd + 1], BELOW_ONE);
		Npp32f nNormalizedV = fminf(src[idd + 2], BELOW_ONE);
	/*	if (id >1000000 && id< 1000010)
		{
			printf("HSV[%u] = [%f, %f, %f]\n", id, nNormalizedH, nNormalizedS, nNormalizedV);
		}*/
		Npp32f nR = 0.0f;
		Npp32f nG = 0.0f;
		Npp32f nB = 0.0f;
		nNormalizedH = nNormalizedH * 6.0F; 

		Npp32f C = nNormalizedS * nNormalizedV;
		Npp32f X = C * fabs( fmodf(1 - nNormalizedH, 2) - 1);
		Npp32f m = nNormalizedV - C;

		Npp32f nI = floorf(nNormalizedH);
		if (nI == 0.0F)
		{
			nR = C; nG = X;
		}
		else if (nI == 1.0F)
		{
			nR = X; nG = C;
		}
		else if (nI == 2.0F)
		{
			nG = C; nB = X;
		}
		else if (nI == 3.0F)
		{
			nG = X; nB = C;
		}
		else if (nI == 4.0F)
		{
			nR = X; nB = C;
		}
		else if (nI == 5.0F)
		{
			nR = C; nB = X;
		}
		dst[idd] = nR + m;
		dst[idd + 1] = nG + m;
		dst[idd + 2] = nB + m;
		
	/*	if (id >1000000 && id< 1000010)
		{
			printf("RGB[%u] = [%f, %f, %f]\n",id, dst[id * 3], dst[id * 3 + 1], dst[id * 3 + 2]);
		}*/

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
		output[index_S] = 0.8f;
		output[index_V] = 0;

		for (size_t i = min_index; i <= max_index; ++i)
		{
			float input_elm = fabsf(input[i * frame_res + id].x);
			output[index_H] += input_elm * omega_arr[i];
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



#include <stdio.h>
#include <cucomplex.h>
#include <cmath>
#include <algorithm>
# include <nppdefs.h>
# include <nppcore.h>
# include <nppi.h>
# include <npps.h>
# include <nppversion.h>
# include <npp.h>
# include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iterator>

void hsv(const cuComplex *input,
	float *output,
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

	

	threshold_top_bottom << <blocks, threads, 0, 0 >> >(tmp_hsv_arr, minH , maxH , frame_res);
	cudaCheckError();
	normalize_frame(tmp_hsv_arr, frame_res); // h
	cudaCheckError();

	gpu_multiply_const(tmp_hsv_arr, frame_res, 0.66f);

	

	gpu_multiply_const(tmp_hsv_arr, frame_res, h);

	//normalize_frame(tmp_hsv_arr + frame_res, frame_res); // s
	//gpu_multiply_const(tmp_hsv_arr + frame_res, frame_res, s);

	normalize_frame(tmp_hsv_arr + frame_res * 2, frame_res); // v
	gpu_multiply_const(tmp_hsv_arr + frame_res * 2, frame_res, v);
	cudaCheckError();

	from_distinct_components_to_interweaved_components << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();

	kernel_normalized_convert_hsv_to_rgb << <blocks, threads, 0, 0 >> > (output, output, frame_res);
	cudaStreamSynchronize(0);
	cudaCheckError();
	
	gpu_multiply_const(output, frame_res * 3, 256.0f);
	cudaStreamSynchronize(0);
	cudaCheckError();

	/*
	static int hhhhh = 0;
	 if(hhhhh < 5000)
	 {
		 hhhhh++;
		 if (hhhhh == 4000)
		 {

			 float* arr = new float[frame_res * 3];

			 cudaMemcpy(arr, output, frame_res * 3 * sizeof(float), cudaMemcpyDeviceToHost);

			 uchar* buffer = new uchar[frame_res * 3];
			 for (size_t i = 0; i < frame_res * 3; i++)
			 {
				 buffer[i] = arr[i];
			 }
			 std::ofstream oo("HSV-test-OUUUUUTT.raw", std::ios::binary);
			 
			 oo.write((char *)buffer, frame_res * 3 * sizeof(char));
			 std::cout << "IMG SAVEEEEDDD&&" << std::endl;
		 }
	 }
		 gpu_multiply_const(output, frame_res * 3, 256.0f);*/

}
