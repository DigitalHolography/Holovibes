
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
/*

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
*/


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

/*
* \brief This function destroys the current image by doing reductions.
*
*/

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


/*
template <unsigned int	blockSize>
__device__ void	warpReduce(volatile	int *sdata, unsigned int  tid) {
	if
		(blockSize >= 64) sdata[tid] += sdata[tid + 32]
		;
	if
		(blockSize >= 32) sdata[tid] += sdata[tid + 16]
		;
	if
		(blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if
		(blockSize >=
			8) sdata[tid] += sdata[tid + 4];
	if
		(blockSize >=
			4) sdata[tid] += sdata[tid + 2];
	if
		(blockSize >=
			2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int	blockSize>
__global__ void	reduce6(int * g_idata, unsigned int n) {
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;

	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		} __syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
		} __syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64];
		} __syncthreads();
	}
	if (tid < 32)
		warpReduce<512>(sdata, tid);
	if (tid == 0)
	{
		g_idata[blockIdx.x] = sdata[0];
	}
}
*/

/*
void normalize_frame_parallel_reduction(float* frame, unsigned int frame_res, float* memory_space)
{
	cudaMemcpy((void **)memory_space, (void **)frame, frame_res * sizeof(float), cudaMemcpyDeviceToDevice);

	//cudaCheckError();
	cudaStreamSynchronize(0);
	float min = -1;
	float max;

	get_minimum_image(memory_space, &min, frame_res);
	cudaStreamSynchronize(0);
	//cudaCheckError();
	printf(" mniniinini is %f\n", min);

}
*/

#define SIZE    10000
/*
void working_sum()
{
	unsigned int threads = 512; //get_max_threads_1d();
	unsigned int 		blocks = map_blocks_to_problem(SIZE, threads);

	int test_arr[SIZE];

	for (unsigned int i = 0; i < SIZE; i++)
	{
		test_arr[i] = 2;
	}

	int* test_gpu_arr;
	cudaMalloc(&test_gpu_arr, sizeof(int) * SIZE);
	cudaMemcpy(test_gpu_arr, test_arr, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

	reduce6<512> << <blocks, threads, threads * sizeof(int) >> > (test_gpu_arr, SIZE);
	cudaMemcpy(test_arr, test_gpu_arr, sizeof(int) * blocks, cudaMemcpyDeviceToHost);

	unsigned long cum = 0;

	for (unsigned i = 0; i < blocks; ++i)
		cum += test_arr[i];

	printf("sum total: %zu\n", cum);


}*/

void min_func_example()
{
	unsigned int threads = 512; //get_max_threads_1d();
	unsigned int 		blocks = map_blocks_to_problem(SIZE, threads);

	float test_arr[SIZE];

	for (unsigned int i = 0; i < SIZE; i++)
	{
		test_arr[i] = 50000.0f - i;
		//printf("%f, ", test_arr[i]);
	}
	printf("\n");
	float* test_gpu_arr;
	float* memspace;
	cudaMalloc(&test_gpu_arr, sizeof(float) * SIZE);
	cudaMalloc(&memspace, sizeof(float) * blocks);

	cudaMemcpy(test_gpu_arr, test_arr, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	float mini = get_minimum_in_image<512>(test_gpu_arr, memspace, SIZE);

	/*
	
	/
	kernel_reduce_min<512> << <blocks, threads, threads * sizeof(float) >> > (test_gpu_arr, test_gpu_arr + SIZE, SIZE);
	cudaMemcpy(test_arr, test_gpu_arr + SIZE, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

	float mini = INFINITY;

	for (unsigned i = 0; i < blocks; ++i)
		mini = std::fmin(mini, test_arr[i]);*/

	printf("min : %f\n", mini);
}

int main()
{
	min_func_example();

	getchar();
	
	return 0;
}


