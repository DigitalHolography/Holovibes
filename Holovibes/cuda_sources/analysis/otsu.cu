#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"
#include "tools_analysis.cuh"
#include "vascular_pulse.cuh"
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
    if (idx < NUM_BINS)
        hist[idx] = 0;
    __syncthreads();

    // Populate shared histogram
    if (idx < imgSize && image[idx] != 0)
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

    return (float)optimalThreshold / (float)NUM_BINS;
}

void compute_binarise_otsu(
    float* input_output, uint* histo_buffer_d, const size_t width, const size_t height, const cudaStream_t stream)
{
    size_t img_size = width * height;

    float global_threshold = otsu_threshold(input_output, histo_buffer_d, img_size, stream);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    global_threshold_kernel<<<blocks, threads, 0, stream>>>(input_output, img_size, global_threshold);
    cudaXStreamSynchronize(stream);
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
    cudaXStreamSynchronize(stream);
}

float get_var_btwclas(const float* var_btwcls, size_t i, size_t j)
{
    size_t idx = (i * (2 * NUM_BINS - i + 1)) / 2 + j - i;
    return var_btwcls[idx];
}

void set_var_btwcls(const float* prob, float* var_btwcls, float* zeroth_moment, float* first_moment)
{
    // Calculer les moments cumulés et remplir var_btwcls
    uint idx;
    float zeroth_moment_ij, first_moment_ij;

    zeroth_moment[0] = prob[0];
    first_moment[0] = prob[0];
    for (uint i = 1; i < NUM_BINS; i++)
    {
        zeroth_moment[i] = zeroth_moment[i - 1] + prob[i];
        first_moment[i] = first_moment[i - 1] + i * prob[i];
        if (zeroth_moment[i] > 0)
            var_btwcls[i] = (first_moment[i] * first_moment[i]) / zeroth_moment[i];
    }
    idx = NUM_BINS;

    for (uint i = 1; i < NUM_BINS; i++)
    {
        for (uint j = i; j < NUM_BINS; j++)
        {

            zeroth_moment_ij = zeroth_moment[j] - zeroth_moment[i - 1];
            if (zeroth_moment_ij > 0)
            {
                first_moment_ij = first_moment[j] - first_moment[i - 1];
                var_btwcls[idx] = (first_moment_ij * first_moment_ij) / zeroth_moment_ij;
            }
            idx += 1;
        }
    }
}

float set_thresh_indices(float* var_btwcls,
                         size_t hist_idx,
                         size_t thresh_idx,
                         size_t thresh_count,
                         float sigma_max,
                         size_t* current_indices,
                         size_t* thresh_indices)
{
    float sigma;
    if (thresh_idx < thresh_count)
    {
        for (uint idx = hist_idx; idx < NUM_BINS - thresh_count + thresh_idx; idx++)
        {
            current_indices[thresh_idx] = idx;
            sigma_max = set_thresh_indices(var_btwcls,
                                           idx + 1,
                                           thresh_idx + 1,
                                           thresh_count,
                                           sigma_max,
                                           current_indices,
                                           thresh_indices);
        }
    }
    else
    {
        sigma = get_var_btwclas(var_btwcls, 0, current_indices[0]) +
                get_var_btwclas(var_btwcls, current_indices[thresh_count - 1] + 1, NUM_BINS - 1);
        for (uint idx = 0; idx < thresh_count - 1; idx++)
        {
            sigma += get_var_btwclas(var_btwcls, current_indices[idx] + 1, current_indices[idx + 1]);
        }
        if (sigma > sigma_max)
        {
            sigma_max = sigma;
            for (size_t i = 0; i < thresh_count; ++i)
            {
                thresh_indices[i] = current_indices[i];
            }
        }
    }

    return sigma_max;
}

__global__ void
histogram_kernel_multi(const float* image, float* hist, float* bin_centers, float minVal, float maxVal, int imgSize)
{
    // Calcul des indices globaux
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcul de la largeur de chaque bin
    float binWidth = (maxVal - minVal) / NUM_BINS;

    // Calcul de l'histogramme
    if (idx < imgSize)
    {
        float pixel = image[idx];
        int binIdx = min(NUM_BINS - 1, int((pixel - minVal) / binWidth));
        atomicAdd(&hist[binIdx], 1); // Utilise atomicAdd pour éviter les conflits
    }

    // Calcule les centres des bins pour un seul thread (thread 0)
    if (idx == 0)
    {
        for (int i = 0; i < NUM_BINS; ++i)
        {
            bin_centers[i] = minVal + (i + 0.5f) * binWidth;
        }
    }
}

void otsu_multi_thresholding(float* input_d,
                             float* histo_buffer_d,
                             float* bin_centers_d,
                             float* thresholds_d,
                             size_t nclasses,
                             size_t size,
                             const cudaStream_t stream)
{
    uint threads = NUM_BINS;
    uint blocks = (size + threads - 1) / threads;

    cudaMemset(histo_buffer_d, 0, NUM_BINS * sizeof(float));
    histogram_kernel_multi<<<blocks, threads>>>(input_d, histo_buffer_d, bin_centers_d, -1, 1, size);
    divide_constant(histo_buffer_d, size, NUM_BINS, stream);
    cudaXStreamSynchronize(stream);

    // Transfer GPU to CPU.
    float hist[NUM_BINS];
    cudaXMemcpy(hist, histo_buffer_d, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);

    int nvalues = count_non_zero<float>(histo_buffer_d, 1, NUM_BINS, stream);
    CHECK(nvalues >= nclasses,
          "otsu_multi_thresholding: At least {} non-zero values needed, got {}",
          nclasses,
          nvalues);

    std::vector<float> thresh(nclasses - 1);
    std::vector<float> bin_center(NUM_BINS);

    cudaXMemcpy(bin_center.data(), bin_centers_d, NUM_BINS * sizeof(float), cudaMemcpyDeviceToHost);

    uint thresh_count = nclasses - 1; // There is n - 1 thresholds.

    if (nvalues == nclasses)
    {
        uint thresh_idx = 0;
        for (uint i = 0; i < NUM_BINS; i++)
        {
            if (thresh_idx == 2)
                break;
            if (hist[i] > 0)
                thresh[thresh_idx++] = static_cast<float>(i);
        }
    }
    else
    {
        size_t thresh_indices[3];
        size_t current_indices[3];
        float var_btwcls[(NUM_BINS * (NUM_BINS + 1) / 2)];
        float zeroth_moment[NUM_BINS];
        float first_moment[NUM_BINS];

        set_var_btwcls(hist, var_btwcls, zeroth_moment, first_moment);
        set_thresh_indices(var_btwcls, 0, 0, thresh_count, 0.5f, current_indices, thresh_indices);

        for (uint i = 0; i < thresh_count; i++)
        {
            thresh[i] = bin_center[thresh_indices[i]];
        }
    }

    // for (int i = 0; i < thresh_count; i++)
    // {
    //     LOG_INFO(i);
    //     LOG_INFO(thresh[i]);
    // }
    cudaXMemcpy(thresholds_d, thresh.data(), thresh_count * sizeof(float), cudaMemcpyHostToDevice);
}

// size_t* thresh_indices;
// size_t* current_indices;

// float* var_btwcls;
// float* zeroth_moment;
// float* first_moment;

// cudaXMalloc(&thresh_indices, thresh_count * sizeof(float));
// cudaXMalloc(&current_indices, thresh_count * sizeof(float));

// cudaXMalloc(&var_btwcls, (NUM_BINS * (NUM_BINS + 1) / 2) * sizeof(float));
// cudaXMalloc(&zeroth_moment, NUM_BINS * sizeof(float));
// cudaXMalloc(&first_moment, NUM_BINS * sizeof(float));

// cudaXMemset(var_btwcls, 0, (NUM_BINS * (NUM_BINS + 1) / 2) * sizeof(float));
// cudaXMemset(zeroth_moment, 0, NUM_BINS * sizeof(float));
// cudaXMemset(first_moment, 0, NUM_BINS * sizeof(float));

// __global__ void
// get_thresholds(float* thresholds_d, const float* bin_centers_d, const size_t* thresh_indices, size_t thresh_count)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < thresh_count)
//         thresholds_d[idx] = bin_centers_d[thresh_indices[idx]];
// }