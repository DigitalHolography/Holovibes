#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"
#include "map.cuh"

using uint = unsigned int;

#define NUM_BINS 256

// Check if optimizable in future with `reduce.cuh` functions.
__global__ void histogram_kernel(const float* image, uint* hist, int imgSize)
{
    extern __shared__ uint shared_hist[]; // Shared memory for histogram bins

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory histogram
    if (tid < NUM_BINS)
        shared_hist[tid] = 0;
    if (idx < NUM_BINS)
        hist[idx] = 0;
    __syncthreads();

    // Populate shared histogram
    if (idx < imgSize)
    {
        int bin = static_cast<int>(image[idx] * (NUM_BINS - 1));
        atomicAdd(shared_hist + bin, 1);
    }
    __syncthreads();

    // Merge shared histograms into global memory
    if (tid < NUM_BINS)
        atomicAdd(&hist[tid], shared_hist[tid]);
}

__global__ void bradley_threshold_kernel(float* output,
                                         const float* input,
                                         int width,
                                         int height,
                                         int windowSize,
                                         float globalThreshold,
                                         float localThresholdFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
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
                localSum += input[j * width + i];
                count++;
            }
        }

        float localMean = localSum / count;
        float localThreshold = localMean * (1 - localThresholdFactor * globalThreshold);
        output[y * width + x] = (input[y * width + x] > localThreshold) * 1.0f;
    }
}

float otsu_threshold(const float* image_d, uint* histo_buffer_d, int size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    size_t shared_mem_size = (NUM_BINS + 1) * sizeof(uint);

    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(image_d, histo_buffer_d, size);

    int histogram[256] = {0};

    cudaXMemcpy(histogram, histo_buffer_d, NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost);

    std::vector<double> probabilities(256, 0.0);
    for (int i = 0; i < 256; ++i)
    {
        probabilities[i] = static_cast<double>(histogram[i]) / size;
    }

    std::vector<double> cumulativeSum(256, 0.0);
    std::vector<double> cumulativeMean(256, 0.0);

    cumulativeSum[0] = probabilities[0];
    cumulativeMean[0] = 0.0;

    for (int i = 1; i < 256; ++i)
    {
        cumulativeSum[i] = cumulativeSum[i - 1] + probabilities[i];
        cumulativeMean[i] = cumulativeMean[i - 1] + i * probabilities[i];
    }

    double globalMean = cumulativeMean[255];

    double maxVariance = 0.0;
    int optimalThreshold = 0;

    for (int t = 0; t < NUM_BINS; ++t)
    {
        if (cumulativeSum[t] == 0 || cumulativeSum[t] == 1)
            continue;

        double betweenClassVariance = std::pow(globalMean * cumulativeSum[t] - cumulativeMean[t], 2) /
                                      (cumulativeSum[t] * (1.0 - cumulativeSum[t]));

        if (betweenClassVariance > maxVariance)
        {
            maxVariance = betweenClassVariance;
            optimalThreshold = t;
        }
    }

    return (float)optimalThreshold / ((float)NUM_BINS - 1.0f);
}

void compute_binarise_otsu(
    float* input_output, uint* histo_buffer_d, const size_t width, const size_t height, const cudaStream_t stream)
{
    size_t img_size = width * height;
    float global_threshold = otsu_threshold(input_output, histo_buffer_d, img_size, stream);
    auto map_function = [global_threshold] __device__(const float input_pixel) -> float
    { return input_pixel > global_threshold; };

    map_generic(input_output, img_size, map_function, stream);
}

void compute_binarise_otsu_bradley(float* output_d,
                                   uint* histo_buffer_d,
                                   const float* input_d,
                                   const size_t width,
                                   const size_t height,
                                   const int window_size,
                                   const float local_threshold_factor,
                                   const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(input_d, histo_buffer_d, img_size, stream);

    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    bradley_threshold_kernel<<<lblocks, lthreads, 0, stream>>>(output_d,
                                                               input_d,
                                                               width,
                                                               height,
                                                               window_size,
                                                               global_threshold,
                                                               local_threshold_factor);
}