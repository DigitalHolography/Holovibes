#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
using uint = unsigned int;

#define NUM_BINS 64

// CUDA kernel to calculate the histogram
__global__ void histogramKernel(float* image, int* hist, int imgSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imgSize)
        atomicAdd(&hist[(unsigned char)(image[idx] * NUM_BINS)], 1);
}

// CUDA kernel to compute Otsu's between-class variance
__global__ void otsuKernel(int* hist, int imgSize, float* betweenClassVariance)
{
    __shared__ int s_hist[NUM_BINS];
    __shared__ float s_prob[NUM_BINS];

    if (int tid = threadIdx.x; tid < NUM_BINS)
    {
        s_hist[tid] = hist[tid];
        s_prob[tid] = (float)s_hist[tid] / imgSize;
    }
    __syncthreads();

    float weightBackground = 0;
    float sumBackground = 0;
    float sumTotal = 0;

    for (int i = 0; i < NUM_BINS; i++)
    {
        sumTotal += i * s_hist[i];
    }

    float maxVariance = 0;
    int optimalThreshold = 0;

    for (int t = 0; t < NUM_BINS; t++)
    {
        weightBackground += s_prob[t];
        float weightForeground = 1.0 - weightBackground;

        if (weightBackground == 0 || weightForeground == 0)
            continue;

        sumBackground += t * s_prob[t];
        float meanBackground = sumBackground / weightBackground;
        float meanForeground = (sumTotal - sumBackground) / weightForeground;

        float varianceBetween =
            weightBackground * weightForeground * (meanBackground - meanForeground) * (meanBackground - meanForeground);
        if (varianceBetween > maxVariance)
        {
            maxVariance = varianceBetween;
            optimalThreshold = t;
        }
    }

    betweenClassVariance[0] = maxVariance;
    betweenClassVariance[1] = (float)optimalThreshold;
}

// CUDA kernel to DO What i want
__global__ void myKernel(float* image, float p, int imgSize)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < imgSize)
        image[index] = ((unsigned char)(image[index] * NUM_BINS) < p) ? 0 : 255;
}

__global__ void myKernel2(float* d_input, float min, float max, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        d_input[tid] = (d_input[tid] - min) / (max - min);
    }
}

void myKernel2_wrapper(float* d_input, float min, float max, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    myKernel2<<<blocks, threads, 0, stream>>>(d_input, min, max, size);
}

// Host function to run Otsu's algorithm using CUDA
void otsuThreshold(float* image, const size_t frame_res, const cudaStream_t stream)
{
    int* d_hist;
    float* d_betweenClassVariance;
    float h_betweenClassVariance[2];

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    // Allocate memory on the GPU
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMalloc(&d_betweenClassVariance, 2 * sizeof(float));

    // Initialize the histogram on the GPU
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    // Run histogram kernel
    histogramKernel<<<blocks, threads, 0, stream>>>(image, d_hist, frame_res); // TODO check 0 befor stram
    cudaDeviceSynchronize();

    // Run Otsu's kernel to compute the optimal threshold
    otsuKernel<<<1, NUM_BINS, 0, stream>>>(d_hist, frame_res, d_betweenClassVariance);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_betweenClassVariance, d_betweenClassVariance, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    //-----------------
    int h_hist[NUM_BINS];

    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < NUM_BINS; i++)
        std::cout << h_hist[i] << " ";
    std::cout << std::endl << "p" << h_betweenClassVariance[1] << std::endl << "-------------------" << std::endl;

    //-----------------
    // TODO
    myKernel<<<blocks, threads, 0, stream>>>(image, h_betweenClassVariance[1], frame_res);
    cudaDeviceSynchronize();

    // Free GPU memory
    cudaFree(d_hist);
    cudaFree(d_betweenClassVariance);

    // return (int)h_betweenClassVariance[1]; // Return optimal threshold
}