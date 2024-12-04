#include "imbinarize.cuh"
#include "cuda_memory.cuh"
#include "vascular_pulse.cuh"
#include "tools_analysis_debug.hh"
#include "compute_env.hh"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <iostream>
#include <cmath>

#define NUM_BINS (1 << 8)

// Check if optimizable in future with `reduce.cuh` functions.
__global__ void histogram_kernel(const float* image, float* hist, int imgSize)
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

__global__ void sumKernel(float* output, const float* input, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ uint sharedData[];

    if (idx < size)
        sharedData[threadIdx.x] = input[idx];
    else
        sharedData[threadIdx.x] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride)
            sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(output, sharedData[0]);
}

void sum(float* output, const float* input, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    sumKernel<<<blocks, threads, threads * sizeof(float)>>>(output, input, size);
    cudaCheckError();
}

__global__ void kernel_cumsum(float* p, float* omega, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        omega[index] = p[index];

        __syncthreads();

        if (index > 0)
            omega[index] += omega[index - 1];
    }
}

void cumsum(float* p, float* omega, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_cumsum<<<blocks, threads, 0, stream>>>(p, omega, size);
    cudaCheckError();
}

void host_cumsum(float* output, float* input, size_t size)
{
    if (size == 0)
        return;

    output[0] = input[0];
    for (size_t i = 1; i < size; ++i)
    {
        output[i] = output[i - 1] + input[i];
    }
}

__global__ void kernel_multiply_with_indices(float* p, float* result, size_t num_bins)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_bins)
        result[index] = p[index] * (index + 1);
}

void multiply_with_indices(float* p, float* result, size_t num_bins, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(num_bins, threads);
    kernel_multiply_with_indices<<<blocks, threads, 0, stream>>>(p, result, num_bins);
    cudaCheckError();
}

__global__ void kernel_sigma_b_squared(float* mu_t, float* omega, float* mu, float* sigma_b_squared, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        float omega_val = omega[index];
        float mu_val = mu[index];

        float diff = *mu_t * omega_val - mu_val;
        sigma_b_squared[index] = (diff * diff) / (omega_val * (1 - omega_val));
    }
}

void compute_sigma_b_squared(
    float* mu_t, float* omega, float* mu, float* sigma_b_squared, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_sigma_b_squared<<<blocks, threads, 0, stream>>>(mu_t, omega, mu, sigma_b_squared, size);
    cudaCheckError();
}

__global__ void kernel_find_max(float* sigma_b_squared, float maxval, float* is_max, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size && sigma_b_squared[index] == maxval)
        is_max[index] = index;
}

void find_mean_index(float* sigma_b_squared, float maxval, float* is_max, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_find_max<<<blocks, threads, 0, stream>>>(sigma_b_squared, maxval, is_max, size);
    cudaCheckError();
}

float mean_non_zero(float* input, int size)
{
    float sum = 0.0f;
    int cpt = 0;

    for (int i = 0; i < size; ++i)
    {
        if (input[i] != 0.0f)
        {
            sum += input[i];
            ++cpt;
        }
    }

    if (cpt > 0)
        return sum / cpt;
    return 0.0f;
}

float otsuthresh(float* counts, holovibes::OtsuStruct& otsu_struct, cudaStream_t stream)
{
    /*
    p = counts / sum(counts);
    omega = cumsum(p);
    mu = cumsum(p .* (1:num_bins)');
    mu_t = mu(end);
    */

    // sum(counts);
    float counts_sum = 0.0f;

    cudaXMemcpyAsync(otsu_struct.d_counts, counts, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaXMemcpyAsync(otsu_struct.d_counts_sum, &counts_sum, sizeof(float), cudaMemcpyHostToDevice, stream);

    size_t d_counts_size = 256;
    sum(otsu_struct.d_counts_sum, otsu_struct.d_counts, d_counts_size, stream);

    // p = counts / sum(counts);
    cudaXMemcpyAsync(otsu_struct.p, counts, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);

    float* h_counts_sum = new float[1];
    cudaXMemcpyAsync(h_counts_sum, otsu_struct.d_counts_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);

    divide_constant(otsu_struct.p, *h_counts_sum, 256 * sizeof(float), stream);

    // omega = cumsum(p);
    float* omega = new float[256];
    float* h_p = new float[256];
    cudaXMemcpy(h_p, otsu_struct.p, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    host_cumsum(omega, h_p, 256);

    // mu = cumsum(p .* (1:num_bins)');
    multiply_with_indices(otsu_struct.p, otsu_struct.p_, 256, stream);

    float* mu = new float[256];
    float* h_p_ = new float[256];
    cudaXMemcpy(h_p_, otsu_struct.p_, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    host_cumsum(mu, h_p_, 256);

    // mu_t = mu(end);
    cudaXStreamSynchronize(stream);
    float mu_tt = mu[255];

    // sigma_b_squared = (mu_t * omega - mu).^2 ./ (omega .* (1 - omega));

    cudaXMemcpyAsync(otsu_struct.d_mu_tt, &mu_tt, sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaXMemcpy(otsu_struct.d_mu, mu, 256 * sizeof(float), cudaMemcpyHostToDevice);

    cudaXMemcpy(otsu_struct.d_omega, omega, 256 * sizeof(float), cudaMemcpyHostToDevice);

    compute_sigma_b_squared(otsu_struct.d_mu_tt,
                            otsu_struct.d_omega,
                            otsu_struct.d_mu,
                            otsu_struct.sigma_b_squared,
                            256 * sizeof(float),
                            stream);

    // maxval = max(sigma_b_squared);

    float* h_sigma_b_squared = new float[256];
    cudaXMemcpy(h_sigma_b_squared, otsu_struct.sigma_b_squared, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    thrust::device_vector<float> d_input(h_sigma_b_squared, h_sigma_b_squared + 256);
    thrust::device_vector<float>::iterator max_it = thrust::max_element(d_input.begin(), d_input.end());
    float maxval = *max_it;

    // idx = mean(find(sigma_b_squared == maxval));

    cudaXMemset(otsu_struct.is_max, 0.0f, sizeof(float) * 256);
    find_mean_index(otsu_struct.sigma_b_squared, maxval, otsu_struct.is_max, 256, stream);

    float* h_is_max = new float[256];
    cudaXMemcpy(h_is_max, otsu_struct.is_max, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    float idx = mean_non_zero(h_is_max, 256);

    // t = (idx - 1 {// 1 indexed in matlab} ) / (num_bins - 1);

    delete[] h_counts_sum;
    delete[] omega;
    delete[] h_p;
    delete[] mu;
    delete[] h_p_;
    delete[] h_sigma_b_squared;
    delete[] h_is_max;

    return idx / (256 - 1);
}

float otsu_compute_threshold(
    float* input, float* histo_buffer_d, const size_t size, holovibes::OtsuStruct& otsu_struct, cudaStream_t stream)
{
    uint threads = NUM_BINS;
    uint blocks = (size + threads - 1) / threads;
    float threshold;
    size_t shared_mem_size = NUM_BINS * sizeof(uint);

    cudaXMemsetAsync(histo_buffer_d, 0, sizeof(uint) * NUM_BINS, stream);

    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(input, histo_buffer_d, size);

    // Histogram is OK, threshold is not

    // otsu_threshold_kernel<<<1, NUM_BINS, 0, stream>>>(histo_buffer_d, size, threshold_d);
    threshold = otsuthresh(histo_buffer_d, otsu_struct, stream);

    return threshold;
}

__global__ void kernel_apply_binarisation(float* input, int size, float globalThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        input[idx] = input[idx] > globalThreshold;
}

void apply_binarisation(
    float* input_output, float threshold, const size_t width, const size_t height, const cudaStream_t stream)
{
    size_t img_size = width * height;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    kernel_apply_binarisation<<<blocks, threads, 0, stream>>>(input_output, img_size, threshold);
    cudaCheckError();
    cudaXStreamSynchronize(stream);
}