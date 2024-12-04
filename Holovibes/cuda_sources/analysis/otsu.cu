#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"

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

__global__ void global_threshold_kernel(float* input, int size, float globalThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        input[idx] = input[idx] > globalThreshold;
}

float otsu_threshold(
    const float* image_d, uint* histo_buffer_d, float* threshold_d, int size, const cudaStream_t stream)
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

    return (float)optimalThreshold / (float)NUM_BINS;
}

void compute_binarise_otsu(float* input_output,
                           uint* histo_buffer_d,
                           float* threshold_d,
                           const size_t width,
                           const size_t height,
                           const cudaStream_t stream)
{
    size_t img_size = width * height;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    global_threshold_kernel<<<blocks, threads, 0, stream>>>(input_output, img_size, threshold);
    cudaXStreamSynchronize(stream);
}
