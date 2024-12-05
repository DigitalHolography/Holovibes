
#include "cuda_memory.cuh"
#include "matrix_operations.hh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

int find_max_thrust(float* input, const size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + size);
    return max_ptr - dev_ptr;
}

int find_min_thrust(float* input, const size_t size)
{
    thrust::device_ptr<float> dev_ptr(input);
    thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr + size);
    return min_ptr - dev_ptr;
}

/*!
 * \brief CUDA kernel to generate a normalized list.
 *
 * This kernel computes values for a normalized list, subtracting the provided limit (`lim`)
 * from the index and storing the result in the output array.
 *
 * \param [out] output Pointer to the output array where results are stored.
 * \param [in] lim The limit value to subtract from each index.
 * \param [in] size The total number of elements to compute.
 */
__global__ void kernel_normalized_list(float* output, int lim, int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = (int)index - lim;
}

/*!
 * \brief Launches a CUDA kernel to generate a normalized list.
 *
 * This function configures and launches the `kernel_normalized_list` on the GPU.
 *
 * \param [out] output Pointer to the output array on the device memory.
 * \param [in] lim The limit value to subtract from each index.
 * \param [in] size The total number of elements to compute.
 * \param [in] stream CUDA stream for asynchronous kernel execution.
 */
void normalized_list(float* output, int lim, int size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_normalized_list<<<blocks, threads, 0, stream>>>(output, lim, size);
    cudaCheckError();
}

/*!
 * \brief Computes the value of the Hermite polynomial of degree `n` at a given point `x`.
 *
 * This device function recursively computes the Hermite polynomial \( H_n(x) \). The computation uses the
 * recurrence relation:
 * \f[
 * H_n(x) = 2xH_{n-1}(x) - 2(n-1)H_{n-2}(x)
 * \f]
 *
 * \param [in] n The degree of the Hermite polynomial (non-negative integer).
 * \param [in] x The point at which to evaluate the polynomial.
 * \return The value of the Hermite polynomial \( H_n(x) \) at \( x \).
 *
 * TODO: This function uses recursion, which can be costly on a GPU. Implement a iterative methods
 * for higher efficiency if performance is critical.
 */
__device__ float comp_hermite(int n, float x)
{
    if (n == 0)
        return 1.0f; // Base case: H_0(x) = 1.
    if (n == 1)
        return 2.0f * x; // Base case: H_1(x) = 2x.
    if (n > 1)
        return (2.0f * x * comp_hermite(n - 1, x)) - (2.0f * (n - 1) * comp_hermite(n - 2, x));
    // Recurrence relation.
    return 0.0f; // This line is a safeguard for invalid input, though n should always be >= 0.
}

/*!
 * \brief Computes the value of a Gaussian function at a given point.
 *
 * This device function evaluates the Gaussian (normal distribution) function:
 * \f[
 * G(x, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}}
 * \f]
 *
 * \param [in] x The point at which to evaluate the Gaussian function.
 * \param [in] sigma The standard deviation (\( \sigma > 0 \)) of the Gaussian distribution.
 * \return The value of the Gaussian function \( G(x, \sigma) \) at \( x \).
 *
 * \note The function assumes that \f$ \sigma > 0 \f$. Passing a non-positive sigma will
 * lead to undefined behavior (e.g., division by zero).
 */
__device__ float comp_gaussian(float x, float sigma)
{
    return 1 / (sigma * (sqrt(2 * M_PI))) * exp((-1 * x * x) / (2 * sigma * sigma));
    // Compute the Gaussian function value.
}

/*!
 * \brief Computes the nth derivative of a Gaussian function at a given point.
 *
 * This device function evaluates the nth derivative of a Gaussian function:
 * \f[
 * G^{(n)}(x, \sigma) = A \cdot H_n\left(\frac{x}{\sigma \sqrt{2}}\right) \cdot G(x, \sigma)
 * \f]
 * where:
 * - \( A = \left(-\frac{1}{\sigma \sqrt{2}}\right)^n \)
 * - \( H_n(x) \) is the nth Hermite polynomial.
 * - \( G(x, \sigma) \) is the Gaussian function.
 *
 * \param [in] x The point at which to evaluate the nth derivative of the Gaussian.
 * \param [in] sigma The standard deviation (\( \sigma > 0 \)) of the Gaussian function.
 * \param [in] n The order of the derivative to compute (\( n \geq 0 \)).
 * \return The value of the nth derivative of the Gaussian function at \( x \).
 *
 * \note This function assumes valid input values for `sigma` and `n`. Specifically,
 *       \( \sigma > 0 \), and \( n \) should be non-negative. Incorrect inputs may
 *       result in undefined behavior.
 */
__device__ float device_comp_dgaussian(float x, float sigma, int n)
{
    float A = pow((-1 / (sigma * sqrt((float)2))), n);       // Coefficient for the nth derivative.
    float B = comp_hermite(n, x / (sigma * sqrt((float)2))); // Hermite polynomial evaluated at normalized x.
    float C = comp_gaussian(x, sigma);                       // Gaussian function value.
    return A * B * C;                                        // Combine components to compute the nth derivative.
}

/*!
 * \brief CUDA kernel to compute the nth derivative of a Gaussian for an array of inputs.
 *
 * This kernel computes the nth derivative of a Gaussian function for each element in the input
 * array and stores the results in the output array. It leverages the `device_comp_dgaussian`
 * function to perform the computation.
 *
 * \param [out] output Pointer to the output array where results are stored (device memory).
 * \param [in] input Pointer to the input array containing the x values (device memory).
 * \param [in] input_size The number of elements in the input array.
 * \param [in] sigma The standard deviation (\( \sigma > 0 \)) of the Gaussian function.
 * \param [in] n The order of the derivative to compute (\( n \geq 0 \)).
 *
 * \note Ensure that the `output` and `input` arrays are allocated in device memory with
 *       sufficient size to store `input_size` elements.
 */
__global__ void kernel_comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
        output[index] = device_comp_dgaussian(input[index], sigma, n); // Compute the nth derivative.
}

/*!
 * \brief Launches a CUDA kernel to compute the nth derivative of a Gaussian function for an array of inputs.
 *
 * This function launches `kernel_comp_dgaussian` to compute the nth derivative of a Gaussian function for each element
 * in the input array. The results are stored in the output array.
 *
 * \param [out] output Pointer to the output array in device memory where results are stored.
 * \param [in] input Pointer to the input array in device memory containing the x values.
 * \param [in] input_size The number of elements in the input array.
 * \param [in] sigma The standard deviation (\( \sigma > 0 \)) of the Gaussian function.
 * \param [in] n The order of the derivative to compute (\( n \geq 0 \)).
 * \param [in] stream The CUDA stream to be used for asynchronous execution.
 *
 * \note Ensure that the `output` and `input` pointers reference valid, allocated memory in the device.
 *       The sizes of both arrays must be at least `input_size` elements.
 */
void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(input_size, threads);
    kernel_comp_dgaussian<<<blocks, threads, 0, stream>>>(output, input, input_size, sigma, n);
    cudaCheckError();
}

/*!
 * \brief CUDA kernel to prepare 2x2 Hessian submatrices.
 *
 * This kernel populates a portion of a Hessian matrix stored in a flat array. For each point
 * indexed by `index`, the kernel calculates the appropriate offset within the output array and
 * assigns a value from the input array `I`. The values are placed at positions determined by the
 * `offset` parameter to construct 2x2 submatrices in the output array.
 *
 * \param [out] output Pointer to the output array in device memory where the Hessian matrices will be stored.
 * \param [in] I Pointer to the input array in device memory containing the source data.
 * \param [in] offset The starting offset in the Hessian matrix array for this computation.
 * \param [in] size The number of elements in the input array to process.
 */
__global__ void kernel_prepare_hessian(float* output, const float* I, const size_t offset, const int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        // Assign input value to the output array at the appropriate offset for the Hessian submatrix.
        output[index * 3 + offset] = I[index];
    }
}

/**
 * \brief Prepares Hessian matrices by launching a CUDA kernel.
 *
 * This function launches the `kernel_prepare_hessian` kernel to compute and populate
 * Hessian matrix entries in the output array.
 *
 * \param [out] output Pointer to the output array in device memory where the Hessian matrices will be stored.
 * \param [in] I Pointer to the input array in device memory containing the source data.
 * \param [in] size The number of elements in the input array to process.
 * \param [in] offset The starting offset in the Hessian matrix array for this computation.
 * \param [in] stream The CUDA stream to be used for asynchronous kernel execution.
 *
 * \note The `output` array must be pre-allocated in device memory with sufficient space to store
 *       the results. The function automatically hardcoded the grid and block dimensions.
 */
void prepare_hessian(float* output, const float* I, const int size, const size_t offset, cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    kernel_prepare_hessian<<<numBlocks, blockSize, 0, stream>>>(output, I, offset, size);
    cudaCheckError();
}

__global__ void kernel_compute_eigen(float* H, int size, float* lambda1, float* lambda2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        double a = H[index * 3], b = H[index * 3 + 1], d = H[index * 3 + 2];
        double trace = a + d;
        double determinant = a * d - b * b;
        double discriminant = trace * trace - 4 * determinant;
        if (discriminant >= 0)
        {
            double eig1 = (trace + std::sqrt(discriminant)) / 2;
            double eig2 = (trace - std::sqrt(discriminant)) / 2;
            if (std::abs(eig1) < std::abs(eig2))
            {
                lambda1[index] = eig1;
                lambda2[index] = eig2;
            }
            else
            {
                lambda1[index] = eig2;
                lambda2[index] = eig1;
            }
        }
    }
}

void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    kernel_compute_eigen<<<blocks, threads, 0, stream>>>(H, size, lambda1, lambda2);
    cudaCheckError();
}

__global__ void
kernel_apply_diaphragm_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared > radius_squared)
            output[index] = 0;
    }
}

void apply_diaphragm_mask(float* output,
                          const float center_X,
                          const float center_Y,
                          const float radius,
                          const short width,
                          const short height,
                          const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_diaphragm_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);
    cudaCheckError();
}

__global__ void
kernel_compute_circle_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared <= radius_squared)
            output[index] = 1;
        else
            output[index] = 0;
    }
}

void compute_circle_mask(float* output,
                         const float center_X,
                         const float center_Y,
                         const float radius,
                         const short width,
                         const short height,
                         const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_compute_circle_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);
    cudaCheckError();
}

__global__ void kernel_apply_mask_and(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
        output[index] *= input[index];
}

void apply_mask_and(float* output, const float* input, const short width, const short height, const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_and<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);
    cudaCheckError();
}

__global__ void kernel_apply_mask_or(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
        output[index] = (input[index] != 0.f) ? 1.f : output[index];
}

void apply_mask_or(float* output, const float* input, const short width, const short height, const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_or<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);
    cudaCheckError();
}

float* compute_gauss_deriviatives_kernel(
    int kernel_width, int kernel_height, float sigma, cublasHandle_t cublas_handler_, cudaStream_t stream)
{
    // Initialize normalized centered at 0 lists, ex for kernel_width = 3 : [-1, 0, 1]
    float* x;
    cudaXMalloc(&x, kernel_width * sizeof(float));
    normalized_list(x, (kernel_width - 1) / 2, kernel_width, stream);

    float* y;
    cudaXMalloc(&y, kernel_height * sizeof(float));
    normalized_list(y, (kernel_height - 1) / 2, kernel_height, stream);

    // Initialize X and Y deriviative gaussian kernels
    float* kernel_x;
    cudaXMalloc(&kernel_x, kernel_width * sizeof(float));
    comp_dgaussian(kernel_x, x, kernel_width, sigma, 2, stream);

    float* kernel_y;
    cudaXMalloc(&kernel_y, kernel_height * sizeof(float));
    comp_dgaussian(kernel_y, y, kernel_height, sigma, 0, stream);

    cudaXStreamSynchronize(stream);

    float* kernel_result;
    cudaXMalloc(&kernel_result, sizeof(float) * kernel_width * kernel_height);
    holovibes::compute::matrix_multiply(kernel_y,
                                        kernel_x,
                                        kernel_height,
                                        kernel_width,
                                        1,
                                        kernel_result,
                                        cublas_handler_,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* result_transpose;
    cudaXMalloc(&result_transpose, sizeof(float) * kernel_width * kernel_height);
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               kernel_width,
                               kernel_height,
                               &alpha,
                               kernel_result,
                               kernel_height,
                               &beta,
                               nullptr,
                               kernel_height,
                               result_transpose,
                               kernel_width));

    // Need to synchronize to avoid freeing too soon
    cudaXStreamSynchronize(stream);
    cudaXFree(kernel_result);

    cudaXFree(x);
    cudaXFree(y);
    cudaXFree(kernel_y);
    cudaXFree(kernel_x);

    return result_transpose;
}

float* compute_kernel(float sigma)
{
    int kernel_size = 2 * std::ceil(2 * sigma) + 1;
    float* kernel = new float[kernel_size * kernel_size];
    float half_size = (kernel_size - 1.0f) / 2.0f;
    float sum = 0.0f;

    int y = 0;
    for (float i = -half_size; i <= half_size; ++i)
    {
        int x = 0;
        for (float j = -half_size; j <= half_size; ++j)
        {
            float value = std::exp(-(i * i + j * j) / (2 * sigma * sigma));

            kernel[x * kernel_size + y] = value;

            sum += value;
            x++;
        }
        y++;
    }

    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

__global__ void kernel_compute_gauss_kernel(float* output, int kernel_size, float sigma, float* d_sum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kernel_size || y >= kernel_size)
        return;

    float half_size = (kernel_size - 1.0f) / 2.0f;
    float i = y - half_size;
    float j = x - half_size;
    float value = expf(-(i * i + j * j) / (2 * sigma * sigma));

    output[y * kernel_size + x] = value;

    // Atomic add to accumulate the total sum (for normalization)
    atomicAdd(d_sum, value);
}

__global__ void kernel_normalize_array(float* input_output, int kernel_size, float* d_sum)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kernel_size || y >= kernel_size)
        return;

    // Normalize each element by the computed sum in d_sum
    input_output[y * kernel_size + x] /= *d_sum;
}

void compute_gauss_kernel(float* output, float sigma, cudaStream_t stream)
{
    float* d_sum;
    float initial_sum = 0.0f;
    int kernel_size = 2 * std::ceil(2 * sigma) + 1;

    // Allocate memory for sum on the device and initialize to 0
    cudaXMalloc(&d_sum, sizeof(float));
    cudaXMemcpyAsync(d_sum, &initial_sum, sizeof(float), cudaMemcpyHostToDevice, stream);

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((kernel_size + blockSize.x - 1) / blockSize.x, (kernel_size + blockSize.y - 1) / blockSize.y);

    // Launch the kernel to compute the Gaussian values
    kernel_compute_gauss_kernel<<<gridSize, blockSize, 0, stream>>>(output, kernel_size, sigma, d_sum);
    cudaCheckError();

    // Normalize the kernel using the computed sum directly on the GPU
    kernel_normalize_array<<<gridSize, blockSize, 0, stream>>>(output, kernel_size, d_sum);
    cudaCheckError();

    // Need to synchronize to avoid freeing too soon
    cudaXStreamSynchronize(stream);
    // Free device memory for sum
    cudaXFree(d_sum);
}

__global__ void kernel_count_non_zero(const float* const input, int* const count, int rows, int cols)
{
    // Shared memory for partial counts
    __shared__ int partial_sum[256];
    int thread_id = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thread_id;
    partial_sum[thread_id] = 0;

    // Check bounds and compute non-zero counts
    if (index < rows * cols && input[index] != 0)
        partial_sum[thread_id] = 1;
    __syncthreads();

    // Reduce within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_id < stride)
            partial_sum[thread_id] += partial_sum[thread_id + stride];
        __syncthreads();
    }

    // Add partial result to global count
    if (thread_id == 0)
        atomicAdd(count, partial_sum[0]);
}

int count_non_zero(const float* const input, const int rows, const int cols, cudaStream_t stream)
{
    int* device_count;
    float* device_input;
    int size = rows * cols;
    int result;

    // Allocate memory on device
    cudaXMalloc((void**)&device_input, size * sizeof(float));
    cudaXMalloc((void**)&device_count, sizeof(int));

    // Copy input matrix to device
    cudaXMemcpyAsync(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Initialize count to 0
    cudaXMemsetAsync(device_count, 0, sizeof(int), stream);

    // Configure kernel
    dim3 threads_per_block(256);
    dim3 blocks_per_grid((size + 255) / 256);

    // Launch kernel
    kernel_count_non_zero<<<blocks_per_grid, threads_per_block, 0, stream>>>(device_input, device_count, rows, cols);
    cudaCheckError();

    // Copy result back to host
    cudaXMemcpyAsync(&result, device_count, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Need to synchronize to avoid freeing too soon
    cudaXStreamSynchronize(stream);
    cudaXFree(device_input);
    cudaXFree(device_count);

    return result;
}

__global__ void
kernel_divide_frames_float_inplace(float* const input_output, const float* const denominator, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = input_output[index] / denominator[index];
        input_output[index] = new_x;
    }
}

void divide_frames_inplace(float* const input_output,
                           const float* const denominator,
                           const uint size,
                           cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_frames_float_inplace<<<blocks, threads, 0, stream>>>(input_output, denominator, size);
    cudaCheckError();
}

// Kernel to normalize an array between a given range
__global__ void
kernel_normalize_array(float* input_output, size_t size, float min_range, float max_range, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        // Normalize to [0, 1], then scale to [min_range, max_range]
        input_output[idx] =
            roundf(((input_output[idx] - min_val) / (max_val - min_val)) * (max_range - min_range) + min_range);
    }
}

// Host function to normalize a device-only array
void normalize_array(float* input_output, size_t size, float min_range, float max_range, cudaStream_t stream)
{
    // Step 1: Use Thrust to find min and max values on the device
    int min_idx = find_min_thrust(input_output, size);
    int max_idx = find_max_thrust(input_output, size);

    // Copy min and max values from device memory to host
    float min_val, max_val;
    cudaXMemcpyAsync(&min_val, input_output + min_idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXMemcpyAsync(&max_val, input_output + max_idx, sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Step 2: Launch kernel to normalize
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);
    kernel_normalize_array<<<blocks, threads, 0, stream>>>(input_output, size, min_range, max_range, min_val, max_val);
    cudaCheckError();
}

void im2uint8(float* image, size_t size, float minVal, float maxVal)
{
    float scale = maxVal - minVal;
    for (size_t i = 0; i < size; ++i)
    {
        float clampedValue = std::max(minVal, std::min(maxVal, image[i]));
        float uint8Value = std::round(255 * (clampedValue - minVal) / scale);
        image[i] = uint8Value;
    }
}
