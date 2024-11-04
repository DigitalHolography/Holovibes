#include "apply_mask.cuh"

#include "hardware_limits.hh"
#include "common.cuh"
#include "complex_utils.cuh"

template <typename T, typename M>
__global__ static void kernel_apply_mask(T* in_out, const M* mask, const size_t size, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            in_out[(size * i) + index] *= mask[index];
        }
    }
}

template <typename T, typename M>
__global__ static void
kernel_apply_mask(const T* input, const M* mask, T* output, const size_t size, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            output[(size * i) + index] = input[(size * i) + index] * mask[index];
        }
    }
}

/*! \brief The CUDA Kernel computing the mean of the pixels inside the image only if the pixel is in the given mask.
 *  This kernel is using shared memory and block reduction.
 *
 *  \param[in] input The input image on which the mask is applied and the mean of pixels is computed.
 *  \param[in] mask The mean will be computed only inside this mask.
 *  \param[in] size The size of the image, e.g : width x height.
 *  \param [in out] mean_vector Vector used to store the sum of the pixels [0] and the number of pixels [1] inside the
 *  circle.
 */
__global__ static void
kernel_get_mean_in_mask(const float* input, const float* mask, const size_t size, float* mean_vector)
{
    extern __shared__ float shared_memory[];          // Getting the shared memory.
    float* shared_sum = shared_memory;                // Pointer to the sum of pixels inside the mask.
    float* shared_count = shared_memory + blockDim.x; // Pointer to the count of pixels inside the mask.

    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint thread_index = threadIdx.x;

    // Setting initial values in the shared memory.
    shared_sum[thread_index] = 0.0f;
    shared_count[thread_index] = 0.0f;

    // Loading data in the shared memory.
    if (index < size && mask[index] != 0)
    {
        shared_sum[thread_index] = input[index];
        shared_count[thread_index] = 1.0f;
    }
    __syncthreads();

    // Hierarchic block reduction in shared memory.
    for (uint stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (thread_index % (2 * stride) == 0)
        {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
            shared_count[thread_index] += shared_count[thread_index + stride];
        }
        __syncthreads();
    }

    // Write block result to global memory.
    // Check for warp level primitive for more optimization.
    if (thread_index == 0)
    {
        atomicAdd(&mean_vector[0], shared_sum[0]);   // Pixels sum
        atomicAdd(&mean_vector[1], shared_count[0]); // Pixels count
    }
}

/*! \brief Cuda kernel performing rescaling operations inside the masks. Set the others pixels to 0.
 *
 *  \param[out] output The output image on which the mask is applied and the pixels are rescaled.
 *  \param[in] input The input image to get the pixels.
 *  \param[in] mask The pixels are rescaled only inside this mask.
 *  \param[in] mean The mean substracted to the pixels.
 *  \param[in] size The size of the image, e.g : width x height.
 */
__global__ static void
kernel_rescale_in_mask(float* output, const float* input, const float* mask, float mean, size_t size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        // If the pixel is inside the mask, the mean is substracted.
        output[index] = (input[index] - mean) * mask[index];
    }
}

template <typename T, typename M>
static void
apply_mask_caller(T* in_out, const M* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_apply_mask<T, M><<<blocks, threads, 0, stream>>>(in_out, mask, size, batch_size);
    cudaCheckError();
}

template <typename T, typename M>
static void apply_mask_caller(
    const T* input, const M* mask, T* output, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_apply_mask<T, M><<<blocks, threads, 0, stream>>>(input, mask, output, size, batch_size);
    cudaCheckError();
}

void apply_mask(
    cuComplex* in_out, const cuComplex* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, cuComplex>(in_out, mask, size, batch_size, stream);
}

void apply_mask(
    cuComplex* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, float>(in_out, mask, size, batch_size, stream);
}

void apply_mask(float* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<float, float>(in_out, mask, size, batch_size, stream);
}

void apply_mask(const cuComplex* input,
                const cuComplex* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, cuComplex>(input, mask, output, size, batch_size, stream);
}

void apply_mask(const cuComplex* input,
                const float* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, float>(input, mask, output, size, batch_size, stream);
}

void apply_mask(const float* input,
                const float* mask,
                float* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<float, float>(input, mask, output, size, batch_size, stream);
}

void get_mean_in_mask(
    const float* input, const float* mask, float* pixels_mean, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    uint shared_memory_size = 2 * threads * sizeof(float);

    // Allocating memory on GPU to compute the sum [0] and number [1] of pixels, used for mean computation.
    float* gpu_mean_vector;
    cudaMalloc(&gpu_mean_vector, 2 * sizeof(float));
    cudaMemset(gpu_mean_vector, 0, 2 * sizeof(float));

    kernel_get_mean_in_mask<<<blocks, threads, shared_memory_size, stream>>>(input, mask, size, gpu_mean_vector);

    // Stream synchronize to make sure all computes are done.
    cudaXStreamSynchronize(stream);

    // Transfering memory to the CPU.
    float cpu_mean_vector[2];
    cudaMemcpy(cpu_mean_vector, gpu_mean_vector, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Avoid zero division
    if (cpu_mean_vector[1] > 0)
        *pixels_mean = cpu_mean_vector[0] / cpu_mean_vector[1];
    else
        *pixels_mean = 0.0f;

    // Release GPU memory
    cudaFree(gpu_mean_vector);

    // Make sur that the mean compute is done.
    cudaCheckError();
}

void rescale_in_mask(
    float* output, const float* input, const float* mask, const float mean, size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_rescale_in_mask<<<blocks, threads, 0, stream>>>(output, input, mask, mean, size);

    cudaCheckError();
}

void rescale_in_mask(float* input_output, const float* mask, const float mean, size_t size, const cudaStream_t stream)
{
    rescale_in_mask(input_output, input_output, mask, mean, size, stream);
}