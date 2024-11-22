#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
using uint = unsigned int;

#define NUM_BINS 256

__global__ void histogram_kernel(float* image, int* hist, int imgSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < imgSize)
        atomicAdd(&hist[(unsigned char)(image[idx] * NUM_BINS)], 1);
}

__global__ void _normalise(float* d_input, float min, float max, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
        d_input[tid] = (int) (((d_input[tid] - min) / (max - min)) * NUM_BINS);
}

void normalise(float* d_input, float min, float max, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    _normalise<<<blocks, threads, 0, stream>>>(d_input, min, max, size);
    cudaDeviceSynchronize();
}

__global__ void global_threshold_kernel(float* input, int size, float globalThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        input[idx] = (input[idx] > globalThreshold) ? 1.0f : 0.0f;
}

__global__ void bradley_threshold_kernel(const float* image,
                                         float* output,
                                         int width,
                                         int height,
                                         int windowSize,
                                         float globalThreshold,
                                         float localThresholdFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x = idx % width;
    int y = idx / width;

    if (x >= width || y >= height)
        return;

    int halfWindow = windowSize / 2;
    int startX = max(x - halfWindow, 0);
    int startY = max(y - halfWindow, 0);
    int endX = min(x + halfWindow, width - 1);
    int endY = min(y + halfWindow, height - 1);

    float localSum = 0;
    int count = 0;

    for (int i = startX; i <= endX; i++)
    {
        for (int j = startY; j <= endY; j++)
        {
            localSum += image[j * width + i];
            count++;
        }
    }

    float localMean = localSum / count;
    float localThreshold = localMean * (1 - localThresholdFactor * globalThreshold);
    output[y * width + x] = (image[y * width + x] > localThreshold) ? 1.0f : 0.0f;
}

float otsu_threshold(float* d_image, int size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    // get histogram
    int h_hist[NUM_BINS];
    int* d_hist;
    cudaMalloc(&d_hist, NUM_BINS * sizeof(int));
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
    histogram_kernel<<<blocks, threads, 0, stream>>>(d_image, d_hist, size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    // Compute optimal threshold
    int total = size;
    float sum = 0, sumB = 0, varMax = 0;
    int wB = 0, wF = 0;
    float threshold = 0;

    for (int i = 0; i < NUM_BINS; i++)
        sum += i * h_hist[i];
    for (int t = 0; t < NUM_BINS; t++)
    {
        wB += h_hist[t];
        if (wB == 0)
            continue;
        wF = total - wB;
        if (wF == 0)
            break;

        sumB += t * h_hist[t];
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = wB * wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax)
        {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold / NUM_BINS;
}

void compute_binarise_otsu(float* d_image, const size_t width, const size_t height, const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(d_image, img_size, stream);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    global_threshold_kernel<<<blocks, threads, 0, stream>>>(d_image, img_size, global_threshold);
    cudaDeviceSynchronize();
}

void compute_binarise_otsu_bradley(float* d_image,
                                   float*& d_output,
                                   const size_t width,
                                   const size_t height,
                                   const int window_size,
                                   const float local_threshold_factor,
                                   const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(d_image, img_size, stream);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    bradley_threshold_kernel<<<blocks, threads, 0, stream>>>(d_image,
                                                             d_output,
                                                             width,
                                                             height,
                                                             window_size,
                                                             global_threshold,
                                                             local_threshold_factor);
    cudaDeviceSynchronize();
}
