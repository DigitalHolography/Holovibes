#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"
#include "tools_analysis_debug.hh"

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
    __syncthreads();

    // Populate shared histogram
    if (idx < imgSize)
    {
        // Clamp les valeurs dans [minVal, maxVal]
        float b = image[idx];
        float clamped_value = max(0.0f, min(1.0f, b));
        // Mise à l'échelle dans [0, 255] et conversion en uint8_t
        float uint8Value = round(255 * clamped_value);
        int bin = static_cast<int>(uint8Value);
        // int bin = static_cast<int>(input[idx] * NUM_BINS);
        atomicAdd(&shared_hist[bin], 1);
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
        input[idx] = (input[idx] > globalThreshold) * 1.0f;
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

__global__ void otsu_threshold_kernel(uint* hist, int total, float* threshold_out)
{
    __shared__ float sum_shared;
    __shared__ float varMax_shared;
    __shared__ float threshold_shared;

    int tid = threadIdx.x;
    if (tid == 0)
    {
        sum_shared = 0;
        varMax_shared = 0;
        threshold_shared = 0;
    }
    __syncthreads();

    // Compute total sum in parallel
    __shared__ float partial_sum[NUM_BINS];
    partial_sum[tid] = (tid < NUM_BINS) ? tid * hist[tid] : 0;
    __syncthreads();

    // Reduce to get total sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
            partial_sum[tid] += partial_sum[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        sum_shared = partial_sum[0];
    __syncthreads();

    // Variables for Otsu
    int wB = 0, wF = 0;
    float sumB = 0;
    float total_sum = sum_shared;

    for (int t = tid; t < NUM_BINS; t += blockDim.x)
    {
        wB += hist[t];
        if (wB == 0)
            continue;

        wF = total - wB;
        if (wF == 0)
            break;

        sumB += t * hist[t];
        float mB = sumB / wB;
        float mF = (total_sum - sumB) / wF;
        float varBetween = wB * wF * (mB - mF) * (mB - mF);

        atomicMax(reinterpret_cast<unsigned int*>(&varMax_shared), __float_as_uint(varBetween));

        if (varBetween == varMax_shared)
            atomicExch(reinterpret_cast<unsigned int*>(&threshold_shared), t);
    }

    __syncthreads();

    if (tid == 0)
        *threshold_out = threshold_shared / NUM_BINS;
}

float otsu_threshold(
    const float* image_d, uint* histo_buffer_d, float* threshold_d, int size, const cudaStream_t stream)
{
    uint threads = NUM_BINS;
    uint blocks = (size + threads - 1) / threads;
    float threshold;
    size_t shared_mem_size = NUM_BINS * sizeof(uint);

    cudaXMemsetAsync(histo_buffer_d, 0, sizeof(uint) * NUM_BINS, stream);

    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(image_d, histo_buffer_d, size);

    // Histogram is OK, threshold is not

    otsu_threshold_kernel<<<1, NUM_BINS, 0, stream>>>(histo_buffer_d, size, threshold_d);

    cudaXStreamSynchronize(stream);

    cudaMemcpy(&threshold, threshold_d, sizeof(float), cudaMemcpyDeviceToHost);

    return threshold;
}

void compute_binarise_otsu(float* input_output,
                           uint* histo_buffer_d,
                           float* threshold_d,
                           const size_t width,
                           const size_t height,
                           const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(input_output, histo_buffer_d, threshold_d, img_size, stream);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    global_threshold_kernel<<<blocks, threads, 0, stream>>>(input_output, img_size, global_threshold);
    std::cout << global_threshold << std::endl;
    cudaXStreamSynchronize(stream);
}

void compute_binarise_otsu_bradley(float* output_d,
                                   uint* histo_buffer_d,
                                   const float* input_d,
                                   float* threshold_d,
                                   const size_t width,
                                   const size_t height,
                                   const int window_size,
                                   const float local_threshold_factor,
                                   const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(input_d, histo_buffer_d, threshold_d, img_size, stream);

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
    cudaXStreamSynchronize(stream);
}
