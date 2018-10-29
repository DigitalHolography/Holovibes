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


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "min_max.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <string>


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

/*
* \brief This function destroys "frame" by doing reductions.
* \param d_frame the image
* \param h_memory_space_sdata space to store results from blocks
*/

template <unsigned int threads>
float get_maximum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res)
{
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

/*
* \brief This function destroys "frame" by doing reductions.
* \param d_frame the image
* \param h_memory_space_sdata space to store results from blocks
*/
template <unsigned int threads>
float get_minimum_in_image(float* d_frame, float* d_memory_space_sdata, unsigned int  frame_res)
{
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

