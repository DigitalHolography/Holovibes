#include "imbinarize.cuh"
#include "cuda_memory.cuh"
#include "vascular_pulse.cuh"
#include "tools_analysis_debug.hh"

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

__global__ void kernel_multiply_with_indices(float* p, float* result, size_t num_bins)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_bins)
    {
        result[index] = p[index] * (index + 1);
    }
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

// Kernel to perform the parallel scan (prefix sum)
__global__ void scan_kernel(float* d_out, float* d_in, int n)
{
    extern __shared__ int temp[]; // Shared memory to hold the data

    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * thid] = d_in[2 * thid];
    temp[2 * thid + 1] = d_in[2 * thid + 1];

    // Up-sweep phase (Reduction in shared memory)
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Down-sweep phase (Calculating prefix sum)
    if (thid == 0)
    {
        temp[n - 1] = 0; // Set the last element to 0
    }

    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // Write results to global memory
    d_out[2 * thid] = temp[2 * thid];
    d_out[2 * thid + 1] = temp[2 * thid + 1];
}

// Function to perform scan (prefix sum) on host
void prefix_sum(float* h_out, float* h_in, int n)
{
    // Launch the kernel with enough threads per block (and shared memory)
    int block_size = 512; // Maximum block size for CUDA
    int grid_size = (n + block_size - 1) / block_size;

    // Execute the scan kernel
    scan_kernel<<<grid_size, block_size, 2 * block_size * sizeof(int)>>>(h_out, h_in, n);
}

float otsuthresh(float* counts, cudaStream_t stream)
{
    /*
    p = counts / sum(counts);
    omega = cumsum(p);
    mu = cumsum(p .* (1:num_bins)');
    mu_t = mu(end);
    */

    // sum(counts);
    float* d_counts;
    float* d_counts_sum;
    float counts_sum = 0.0f;

    cudaXMallocAsync(&d_counts, 256 * sizeof(float), stream);
    cudaXMallocAsync(&d_counts_sum, sizeof(float), stream);
    cudaXMemcpyAsync(d_counts, counts, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaXMemcpyAsync(d_counts_sum, &counts_sum, sizeof(float), cudaMemcpyHostToDevice, stream);

    size_t d_counts_size = 256;
    sum(d_counts_sum, d_counts, d_counts_size, stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(d_counts_sum, 1, 1, "d_counts_sum", stream);

    // p = counts / sum(counts);
    float* p;
    cudaXMallocAsync(&p, 256 * sizeof(float), stream);
    cudaXMemcpyAsync(p, counts, 256 * sizeof(float), cudaMemcpyHostToDevice, stream);

    float* h_counts_sum = new float[1];
    cudaXMemcpyAsync(h_counts_sum, d_counts_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);

    divide_constant(p, *h_counts_sum, 256 * sizeof(float), stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(p, 256, 1, "p", stream);

    // omega = cumsum(p);
    float* omega;
    cudaXMallocAsync(&omega, 256 * sizeof(float), stream);

    prefix_sum(omega, p, 256 * sizeof(float));
    // cumsum(p, omega, 256 * sizeof(float), stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(omega, 256, 1, "omega", stream);

    // mu = cumsum(p .* (1:num_bins)');
    float* p_;
    cudaXMallocAsync(&p_, 256 * sizeof(float), stream);

    multiply_with_indices(p, p_, 256, stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(p_, 256, 1, "p_", stream);

    float* mu;
    cudaXMallocAsync(&mu, 256 * sizeof(float), stream);

    cumsum(p_, mu, 256 * sizeof(float), stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(mu, 256, 1, "mu", stream);

    // mu_t = mu(end);
    float* h_mu = new float[256];
    cudaXMemcpyAsync(h_mu, mu, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    float mu_tt = h_mu[255];

    // sigma_b_squared = (mu_t * omega - mu).^2 ./ (omega .* (1 - omega));
    float* sigma_b_squared;
    cudaXMallocAsync(&sigma_b_squared, 256 * sizeof(float), stream);

    float* d_mu_tt;
    cudaXMallocAsync(&d_mu_tt, sizeof(float), stream);
    cudaXMemcpyAsync(d_mu_tt, &mu_tt, sizeof(float), cudaMemcpyHostToDevice, stream);

    compute_sigma_b_squared(d_mu_tt, omega, mu, sigma_b_squared, 256 * sizeof(float), stream);

    cudaXStreamSynchronize(stream);
    print_in_file_gpu(sigma_b_squared, 256, 1, "sigma_b_squared", stream);

    return 0.0f;
}

float otsu_compute(float* input, float* histo_buffer_d, const size_t size, cudaStream_t stream)
{
    uint threads = NUM_BINS;
    uint blocks = (size + threads - 1) / threads;
    float threshold;
    size_t shared_mem_size = NUM_BINS * sizeof(uint);

    cudaXMemsetAsync(histo_buffer_d, 0, sizeof(uint) * NUM_BINS, stream);

    histogram_kernel<<<blocks, threads, shared_mem_size, stream>>>(input, histo_buffer_d, size);

    // Histogram is OK, threshold is not

    // otsu_threshold_kernel<<<1, NUM_BINS, 0, stream>>>(histo_buffer_d, size, threshold_d);
    threshold = otsuthresh(histo_buffer_d, stream);

    return threshold;
}