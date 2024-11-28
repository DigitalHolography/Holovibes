#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"

#include <cmath>

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
        int bin = static_cast<int>(image[idx] * NUM_BINS);
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

    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(image_d, histo_buffer_d, size);

    otsu_threshold_kernel<<<1, NUM_BINS, 0, stream>>>(histo_buffer_d, size, threshold_d);

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

/*
Function OtsuMultiThresholding(image, num_thresholds):
    Input: image (grayscale), num_thresholds (number of thresholds)
    Output: thresholds (list of threshold values)

    # Step 1: Compute the histogram of the image
    histogram = ComputeHistogram(image)  # array of size 256 for 8-bit images
    total_pixels = sum(histogram)  # total number of pixels in the image

    # Step 2: Normalize the histogram to get probabilities
    probabilities = [histogram[i] / total_pixels for i in range(256)]

    # Step 3: Initialize variables
    best_thresholds = []  # list to store optimal thresholds
    min_within_class_variance = Infinity  # track minimum intra-class variance

    # Step 4: Generate all possible combinations of thresholds
    all_combinations = GenerateThresholdCombinations(range(256), num_thresholds)

    For each combination of thresholds in all_combinations:
        # Step 5: Divide the histogram into (num_thresholds + 1) classes
        thresholds = [t1, t2, ..., tn]  # current combination of thresholds
        class_ranges = DivideIntoClasses(probabilities, thresholds)

        # Step 6: Compute within-class variance for the current combination
        within_class_variance = 0
        For each class_range in class_ranges:
            weight = Sum(class_range)  # total probability (weight) of the class
            mean = WeightedMean(class_range)  # mean intensity of the class
            variance = WeightedVariance(class_range, mean)  # variance of the class
            within_class_variance += weight * variance

        # Step 7: Update optimal thresholds if variance is minimized
        If within_class_variance < min_within_class_variance:
            min_within_class_variance = within_class_variance
            best_thresholds = thresholds

    Return best_thresholds
*/

float get_var_btwclas(const std::vector<float>& var_btwcls, size_t i, size_t j)
{
    uint idx = (i * (2 * NUM_BINS - i + 1)) / 2 + j - i;
    return var_btwcls[idx];
}

void set_var_btwcls(const std::vector<float>& prob,
                    std::vector<float>& var_btwcls,
                    std::vector<float>& zeroth_moment,
                    std::vector<float>& first_moment)
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

float set_thresh_indices(std::vector<float>& var_btwcls,
                         size_t hist_idx,
                         size_t thresh_idx,
                         size_t thresh_count,
                         float sigma_max,
                         std::vector<size_t>& current_indices,
                         std::vector<size_t>& thresh_indices)
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
            for (size_t i = 0; i < thresh_indices.size(); ++i)
            {
                thresh_indices[i] = current_indices[i];
            }
        }
    }

    return sigma_max;
}

// void threshold_multiotsu(image = None, classes = 3, nbins = 256, *, hist = None)

__global__ void rescale_csv_kernel(float* output, const float* input, size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Transformation : [-1, 1] -> [0, 255]
        output[idx] = (input[idx] + 1.0f) / 2; //* 127.5f;
    }
}

__global__ void histogram_kernel_multi(const float* image, uint* hist, int imgSize)
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
        int bin = static_cast<int>(image[idx]);
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();

    // Merge shared histograms into global memory
    if (tid < NUM_BINS)
        atomicAdd(&hist[tid], shared_hist[tid]);
}

void otsu_multi_thresholding(const float* input_d,
                             float* otsu_rescale,
                             uint* histo_buffer_d,
                             float* thresholds_d,
                             size_t nclasses,
                             size_t size,
                             const cudaStream_t stream)
{

    // Step 1 : Compute the histogram of the image
    uint threads = NUM_BINS;
    uint blocks = (size + threads - 1) / threads;
    size_t shared_mem_size = NUM_BINS * sizeof(uint);

    rescale_csv_kernel<<<blocks, threads, 0, stream>>>(otsu_rescale, input_d, size);

    // cudaXMalloc(&hist, NUM_BINS * sizeof(uint));
    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(otsu_rescale, histo_buffer_d, size);
    // histogram_kernel_multi<<<blocks, threads, shared_mem_size, stream>>>(input_d, histo_buffer_d, size);
    cudaXStreamSynchronize(stream);

    // Transferer GPU TO CPU.
    uint hist[NUM_BINS];
    // cudaXMalloc(&hist, NUM_BINS * sizeof(uint));
    cudaXMemcpy(hist, histo_buffer_d, NUM_BINS * sizeof(uint), cudaMemcpyDeviceToHost);

    uint nvalues = 0;
    for (uint i = 0; i < NUM_BINS; i++)
    {
        if (hist[i] > 0)
            nvalues++;
    }

    CHECK(nvalues >= nclasses, "NIK ZEBI");

    std::vector<float> prob(NUM_BINS);
    for (uint i = 0; i < NUM_BINS; i++)
    {
        prob[i] = static_cast<float>(hist[i]);
    }

    std::vector<uint> thresh(nclasses - 1);
    std::vector<uint> bin_center(NUM_BINS);
    for (uint i = 0; i < NUM_BINS; i++)
        bin_center[i] = i;

    if (nvalues == nclasses)
    {
        uint thresh_idx = 0;
        for (uint i = 0; i < NUM_BINS; i++)
        {
            if (thresh_idx == 2)
                break;
            if (prob[i] > 0)
                thresh[thresh_idx++] = i;
        }
    }
    else
    {
        uint thresh_count = nclasses - 1; // classes = 3 mais on fait classes-1.

        std::vector<size_t> thresh_indices(thresh_count);
        std::vector<size_t> current_indices(thresh_count);
        std::vector<float> var_btwcls(NUM_BINS * (NUM_BINS + 1) / 2, 0.0f);
        std::vector<float> zeroth_moment(NUM_BINS, 0.0f);
        std::vector<float> first_moment(NUM_BINS, 0.0f);

        set_var_btwcls(prob, var_btwcls, zeroth_moment, first_moment);
        set_thresh_indices(var_btwcls, 0, 0, thresh_count, 0.0f, current_indices, thresh_indices);

        /* le res est tresh indices*/

        for (uint i = 0; i < thresh_count; i++)
        {
            thresh[i] = bin_center[thresh_indices[i]];
        }
    }

    for (int i = 0; i < nclasses - 1; i++)
    {
        LOG_INFO(i);
        LOG_INFO((static_cast<float>(thresh[i]) / 255) * 2 - 1);
    }
}

// float* proba = new float(NUM_BINS);

// // Step 2 : Normalize the histogram to get probabilities
// for (size_t i = 0; i < NUM_BINS; i++)
//     proba[i] = histo[i] / size;

// // Step 3 : Initialize variables
// float* best_threshold = new float(nb_thresholds);
// float min_within_class_variance = HUGE_VALF;

// // Step 4 : Generate all possible combinations of thresholds
// // all_combinations = GenerateThresholdCombinations(range(256), num_thresholds)

// uint* all_combinations = new uint(3 * NUM_BINS * NUM_BINS * NUM_BINS);
