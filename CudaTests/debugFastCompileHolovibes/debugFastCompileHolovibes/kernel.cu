
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <string>


unsigned int get_max_threads_1d()
{
	static int max_threads_per_block_1d;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_threads_per_block_1d = prop.maxThreadsPerBlock;
		initialized = true;
	}

	return max_threads_per_block_1d;
}



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
	// message erreur

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
void kernel_reduce_min(float* g_idata, float* g_odata, unsigned int frame_res) {
	extern __shared__ float sdata_min[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata_min[tid] = INFINITY;

	while (i < frame_res){
		float tmp_min = g_idata[i];
		if (i + blockSize < frame_res)
			tmp_min = fminf(tmp_min, g_idata[i + blockSize]);
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
		g_odata[blockIdx.x] = sdata_min[0];
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
void kernel_reduce_max(float* g_idata, float* g_odata, unsigned int frame_res) {
	extern __shared__ float sdata_max[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata_max[tid] = - 1;

	while (i < frame_res) {
		float tmp_min = g_idata[i];
		if (i + blockSize < frame_res)
			tmp_min = fmaxf(tmp_min, g_idata[i + blockSize]);
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
		g_odata[blockIdx.x] = sdata_max[0];
}

/*
* \brief This function destroys the current image by doing reductions.
*
*/

template <unsigned int threads>
float get_maximum_in_image(float* frame, float* memory_space_sdata, unsigned int  frame_res)
{
	unsigned int blocks = map_blocks_to_problem(frame_res, threads);

	kernel_reduce_max<threads> << <blocks, threads, threads * sizeof(float) >> > (frame, memory_space_sdata, frame_res);

	float *result_array = new float[blocks];

	cudaMemcpy(result_array, memory_space_sdata, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	float result = - 1;

	for (unsigned i = 0; i < blocks; ++i)
		result = std::fmax(result, result_array[i]);

	delete result_array;

	return result;
}

template <unsigned int threads>
float get_minimum_in_image(float* frame, float* memory_space_sdata, unsigned int  frame_res)
{
	unsigned int blocks = map_blocks_to_problem(frame_res, threads);

	kernel_reduce_min<threads> << <blocks, threads, threads * sizeof(float) >> > (frame, memory_space_sdata, frame_res);

	float *result_array = new float[blocks];

	cudaMemcpy(result_array, memory_space_sdata, blocks * sizeof(float), cudaMemcpyDeviceToHost);

	float result = INFINITY;

	for (unsigned i = 0; i < blocks; ++i) {
		

		result = std::fmin(result, result_array[i]);
//		printf("%f, ", result_array[i]);
	}
	printf("%f, ", result);


	delete result_array;

	return result;
}


#define SIZE    10000


void min_func_example()
{
	unsigned int threads = 512; //get_max_threads_1d();
	unsigned int 		blocks = map_blocks_to_problem(SIZE, threads);

	float test_arr[SIZE];

	for (unsigned int i = 0; i < SIZE; i++)
	{
		test_arr[i] = (50000 + i) % 100;
	}
	
	float* test_gpu_arr;
	float* memspace;
	cudaMalloc(&test_gpu_arr, sizeof(float) * SIZE);
	cudaMalloc(&memspace, sizeof(float) * blocks);

	cudaMemcpy(test_gpu_arr, test_arr, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	float maxi = get_maximum_in_image<512>(test_gpu_arr, memspace, SIZE);

	printf("max : %f\n", maxi);
}

int main()
{
	min_func_example();

	getchar();
	
	return 0;
}


