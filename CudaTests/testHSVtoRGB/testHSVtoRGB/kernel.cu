
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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



# define SAMPLING_FREQUENCY  1

using uint = unsigned int;

unsigned int get_max_blocks()
{
	static int max_blocks;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_blocks = prop.maxGridSize[0];
		initialized = true;
	}

	return max_blocks;
}

inline unsigned map_blocks_to_problem(const size_t problem_size,
	const unsigned nb_threads)
{
	unsigned nb_blocks = static_cast<unsigned>(
		std::ceil(static_cast<float>(problem_size) / static_cast<float>(nb_threads)));

	if (nb_blocks > get_max_blocks())
		nb_blocks = get_max_blocks();

	return nb_blocks;
}


template <unsigned int blockSize>__device__
void kernel_warp_reduce_min(volatile float* sdata_min, unsigned int tid) {
	if (blockSize >= 64)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 32]);
	if (blockSize >= 32)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 16]);
	if (blockSize >= 16)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 8]);
	if (blockSize >= 8)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 4]);
	if (blockSize >= 4)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 2]);
	if (blockSize >= 2)
		sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 1]);
}



template <unsigned int blockSize> __global__
void kernel_reduce_min(float* d_frame, float* d_memory_space_sdata, unsigned int frame_res) {
	extern __shared__ float sdata_min[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata_min[tid] = INFINITY;

	while (i < frame_res) {
		float tmp_min = d_frame[i];
		if (i + blockSize < frame_res)
			tmp_min = fminf(tmp_min, d_frame[i + blockSize]);
		sdata_min[tid] = fminf(sdata_min[tid], tmp_min);
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + 64]);
		}
		__syncthreads();
	}
	if (tid < 32)
		kernel_warp_reduce_min<blockSize>(sdata_min, tid);
	if (tid == 0)
		d_memory_space_sdata[blockIdx.x] = sdata_min[0];
}



template <unsigned int blockSize>__device__
void kernel_warp_reduce_max(volatile float* sdata_max, unsigned int tid) {
	if (blockSize >= 64)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 32]);
	if (blockSize >= 32)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 16]);
	if (blockSize >= 16)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 8]);
	if (blockSize >= 8)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 4]);
	if (blockSize >= 4)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 2]);
	if (blockSize >= 2)
		sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 1]);
}


template <unsigned int blockSize> __global__
void kernel_reduce_max(float* d_frame, float* d_memory_space_sdata, unsigned int frame_res) {
	extern __shared__ float sdata_max[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata_max[tid] = -1;

	while (i < frame_res) {
		float tmp_min = d_frame[i];
		if (i + blockSize < frame_res)
			tmp_min = fmaxf(tmp_min, d_frame[i + blockSize]);
		sdata_max[tid] = fmaxf(sdata_max[tid], tmp_min);
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + 64]);
		}
		__syncthreads();
	}
	if (tid < 32)
		kernel_warp_reduce_max<blockSize>(sdata_max, tid);
	if (tid == 0)
		d_memory_space_sdata[blockIdx.x] = sdata_max[0];
}

float get_maximum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res)
{
	unsigned const threads = 512;
	unsigned int blocks = map_blocks_to_problem(frame_res, threads);
	kernel_reduce_max<threads> << <blocks, threads, threads * sizeof(float) >> > (d_frame, d_memory_space_sdata, frame_res);
	float *h_result_array = new float[blocks];
	cudaMemcpy(h_result_array, d_memory_space_sdata, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	float result = -1;
	for (unsigned i = 0; i < blocks; ++i)
		result = std::fmax(result, h_result_array[i]);
	delete[] h_result_array;
	return result;
}


float get_minimum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res)
{
	unsigned const threads = 512;
	unsigned int blocks = map_blocks_to_problem(frame_res, threads);
	kernel_reduce_min<threads> << <blocks, threads, threads * sizeof(float) >> > (d_frame, d_memory_space_sdata, frame_res);
	float *result_array = new float[blocks];
	cudaMemcpy(result_array, d_memory_space_sdata, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	float result = INFINITY;

	for (unsigned i = 0; i < blocks; ++i)
		result = std::fmin(result, result_array[i]);

	delete[] result_array;
	return result;
}

void get_minimum_maximum_in_image(const float *frame, const unsigned frame_res, float* min, float* max)
{
	const uint threads = 512;
	const uint blocks = map_blocks_to_problem(frame_res, threads);


	float *d_tmp_storage;
	cudaMalloc(&d_tmp_storage, sizeof(float) * frame_res + sizeof(float) * blocks);

	cudaMemcpy(d_tmp_storage, frame, sizeof(float) * frame_res, cudaMemcpyDeviceToDevice);

	*min = get_minimum_in_image(d_tmp_storage, d_tmp_storage + frame_res, frame_res);

	cudaMemcpy(d_tmp_storage, frame, sizeof(float) * frame_res, cudaMemcpyDeviceToDevice);

	*max = get_maximum_in_image(d_tmp_storage, d_tmp_storage + frame_res, frame_res);

	cudaFree(d_tmp_storage);
}


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

		Npp32f nNormalizedH = fminf(src[idd], 0.999999f);
		Npp32f nNormalizedS = fminf(src[idd + 1], 0.9999999f);
		Npp32f nNormalizedV = fminf(src[idd + 2], 0.9999999f);
/*
		if (id < 10)
		{
			printf("HSV[%u] = [%f, %f, %f]\n", id, nNormalizedH, nNormalizedS, nNormalizedV);
		}*/

		Npp32f nR = 0.0f;
		Npp32f nG = 0.0f;
		Npp32f nB = 0.0f;
		nNormalizedH = nNormalizedH * 6.0F;

		Npp32f C = nNormalizedS * nNormalizedV;
		Npp32f X = C * fabs(fmodf(1 - nNormalizedH, 2) - 1);
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
		/*
		if (id < 10)
		{
			printf("RGB[%u] = [%f, %f, %f]\n", id, dst[idd], dst[idd + 1], dst[idd + 2]);
		}
*/
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
			//output[index_S] += input_elm * omega_arr[omega_size + i];
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

__global__
void kernel_multiply_frames_float(const float	*input1,
	const float		*input2,
	float			*output,
	const uint		size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	output[index] = input1[index] * input2[index];
}

__global__
void kernel_multiply_const(float		*frame,
	uint		frame_size,
	float		x)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_size)
		frame[id] *= x;
}

void gpu_multiply_const(float		*frame,
	uint		frame_size,
	float		x)
{
	uint		threads = 512;
	uint		blocks = map_blocks_to_problem(frame_size, threads);
	kernel_multiply_const << <blocks, threads, 0, 0 >> > (frame, frame_size, x);
}


__global__
void kernel_substract_const(float		*frame,
	uint		frame_size,
	float		x)
{
	const uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_size)
		frame[id] -= x;
}

void gpu_substract_const(float		*frame,
	uint		frame_size,
	float		x)
{
	uint		threads = 512;
	uint		blocks = map_blocks_to_problem(frame_size, threads);
	kernel_substract_const << <blocks, threads, 0, 0 >> > (frame, frame_size, x);
}

void normalize_frame(float* frame, uint frame_res)
{
	float min, max;

	get_minimum_maximum_in_image(frame, frame_res, &min, &max);

	
	gpu_substract_const(frame, frame_res, min);
	cudaStreamSynchronize(0);
	gpu_multiply_const(frame, frame_res, 1 / (max - min )); // need to be below 1 for some reason
	cudaStreamSynchronize(0);
}


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

	const uint threads = 512;
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
		}
		cudaMalloc(&tmp_hsv_arr, sizeof(float) * frame_res * 3 * 2); // HSV array * 2 , second part is for parallel reduction
	}

	if (omega_arr_size != nb_img)
	{
		omega_arr_size = nb_img;
		if (omega_arr_data)
		{
			cudaFree(omega_arr_data);
		}

		cudaMalloc(&omega_arr_data, sizeof(float) * nb_img * 2); // w1[] && w2[]

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
	}

	kernel_compute_and_fill_hsv << <blocks, threads, 0, 0 >> > (input, output, frame_res,
		index_min, index_max, index_max - index_min + 1, omega_arr_size, omega_arr_data);
	cudaStreamSynchronize(0);

	from_interweaved_components_to_distinct_components << <blocks, threads, 0, 0 >> > (output, tmp_hsv_arr, frame_res);
	cudaStreamSynchronize(0);


	normalize_frame(tmp_hsv_arr, frame_res); // h

	gpu_multiply_const(tmp_hsv_arr, frame_res, h);

	threshold_top_bottom << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, minH * h, maxH * h, frame_res);
	normalize_frame(tmp_hsv_arr, frame_res); // h

	gpu_multiply_const(tmp_hsv_arr, frame_res, h);

	//normalize_frame(tmp_hsv_arr + frame_res, frame_res); // s
	//gpu_multiply_const(tmp_hsv_arr + frame_res, frame_res, s);

	normalize_frame(tmp_hsv_arr + frame_res * 2, frame_res); // v
	gpu_multiply_const(tmp_hsv_arr + frame_res * 2, frame_res, v);

	from_distinct_components_to_interweaved_components << <blocks, threads, 0, 0 >> > (tmp_hsv_arr, output, frame_res);
	cudaStreamSynchronize(0);
	/*
	gpu_multiply_const(output, frame_res * 3, 255);
	cudaCheckError();

	Npp8u* d_uint_space;
	cudaMalloc(&d_uint_space, sizeof(Npp8u) * frame_res * 3 * 2); // 2 hsv arrays, 1: in, 2: out
	Npp8u* d_uint_space_out = d_uint_space + frame_res * 3;

	float_to_uint8(output, d_uint_space, frame_res * 3);

	NppiSize dimension = {width, height};
	nppiHSVToRGB_8u_C3R(d_uint_space, width, d_uint_space_out, width, dimension);
	//convert_hsv_to_rgb_255 << <blocks, threads, 0, 0 >> > (d_uint_space, d_uint_space_out, frame_res);

	uint8_to_float(d_uint_space_out, output, frame_res * 3);
	cudaFree(d_uint_space);
	*/

	kernel_normalized_convert_hsv_to_rgb << <blocks, threads, 0, 0 >> > (output, output, frame_res);
	cudaStreamSynchronize(0);

	gpu_multiply_const(output, frame_res * 3, 255 * 255);
	cudaStreamSynchronize(0);

}

#define SIZEARR  750000

void open_image_to_test()
{
	using uchar = unsigned char;

	const uint threads = 512;
	uint blocks = map_blocks_to_problem(500 * 500, threads);
	std::ifstream input("HSV-test.raw", std::ios::binary);
	uchar* buffer = new uchar[SIZEARR];
	input.read((char*)buffer, SIZEARR);
	input.seekg(0, std::ios::beg);
	
	float* arr = new float[SIZEARR];

	for (size_t i = 0; i < SIZEARR; i++)
	{
		arr[i] = buffer[i];
	}

	/*
	for (size_t i = 0; i < 500; i++)
	{
		for (size_t j = 0; j < 500; j++)
		{
			unsigned index = 3 * (i * 500 + j);
			std::cout << "[" << i << "," << j << "]";
			std::cout << "(" << arr[index] << "," << arr[index + 1] << "," << arr[index + 2] << ")";
			std::cout << std::endl;
		}
	}*/
	
	float* d_arr;
	

	unsigned frame_res = SIZEARR / 3;

	cudaMalloc(&d_arr, SIZEARR * sizeof(float) * 2);
	cudaMemcpy(d_arr, arr, SIZEARR * sizeof(float), cudaMemcpyHostToDevice);
	float* d_arr2 = d_arr + SIZEARR;
	from_interweaved_components_to_distinct_components << <blocks, threads, 0, 0 >> > (d_arr, d_arr2, frame_res);
	cudaStreamSynchronize(0);

	//gpu_multiply_const(d_arr2, SIZEARR, 1 / 256.0f);


	
	normalize_frame(d_arr2 , frame_res); // h

	normalize_frame(d_arr2 + frame_res, frame_res); // s

	normalize_frame(d_arr2 + frame_res * 2, frame_res); // v

	
	

	from_distinct_components_to_interweaved_components << <blocks, threads, 0, 0 >> > (d_arr2, d_arr, frame_res);
	cudaStreamSynchronize(0);



	kernel_normalized_convert_hsv_to_rgb << <blocks, threads, 0, 0 >> > (d_arr, d_arr, frame_res);
	gpu_multiply_const(d_arr, SIZEARR, 256.0f);

	cudaMemcpy(arr,d_arr, SIZEARR * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < SIZEARR; i++)
	{
		 buffer[i] = arr[i];
	}

	
	std::ofstream output("HSV-test-OUUUUUTT.raw", std::ios::binary);


	output.write((char *)buffer, SIZEARR * sizeof(char));
	/*std::copy(
		std::istreambuf_iterator<char>(input),
		std::istreambuf_iterator<char>(),
		std::ostreambuf_iterator<char>(output));*/
}

int main()
{
	open_image_to_test();

//	getchar();
	return 0;
}