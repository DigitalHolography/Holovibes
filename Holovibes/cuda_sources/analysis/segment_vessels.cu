#include "cuda_memory.cuh"
#include "map.cuh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

__global__ void kernel_minus_negation_times_2(float* const input_output, const float* const input, const size_t size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        input_output[index] -= !input[index] * 2;
}

void minus_negation_times_2(float* const R_vascular_pulse,
                            const float* const mask_vesselnessClean,
                            const size_t size,
                            const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_minus_negation_times_2<<<blocks, threads, 0, stream>>>(R_vascular_pulse, mask_vesselnessClean, size);
    cudaCheckError();
}

void negation(float* const input_output, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    auto map_function = [] __device__(const float input_pixel) -> float { return !input_pixel; };
    map_generic(input_output, size, map_function, stream);
}

__global__ void kernel_quantize(float* const output,
                                const float* const input,
                                const float* const thresholds,
                                const int length_input,
                                const int lenght_threshold)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < length_input)
    {
        float value = input[index];
        int quantized_level = 1;

        for (int t = 0; t < lenght_threshold; ++t)
        {
            if (value > thresholds[t])
                quantized_level = t + 2;
            else
                break;
        }

        output[index] = quantized_level;
    }
}

void imquantize(float* const output,
                const float* const input,
                const float* const thresholds,
                const size_t length_input,
                const size_t lenght_threshold,
                const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(length_input, threads);
    kernel_quantize<<<blocks, threads, 0, stream>>>(output, input, thresholds, length_input, lenght_threshold);
    cudaCheckError();
}

void segment_vessels(float* const output,
                     float* const new_thresholds,
                     float* const R_VascularPulse,
                     const float* const mask_vesselness_clean,
                     const size_t size,
                     const float* thresholds,
                     const cudaStream_t stream)
{
    float minus_one = -1;
    cudaXMemcpyAsync(new_thresholds + 1, thresholds, sizeof(float) * 3, cudaMemcpyHostToDevice, stream);
    cudaXMemcpyAsync(new_thresholds, &minus_one, sizeof(float), cudaMemcpyHostToDevice, stream);

    minus_negation_times_2(R_VascularPulse, mask_vesselness_clean, size, stream);
    imquantize(output, R_VascularPulse, new_thresholds, size, 4, stream);
}

void is_equal_to_either(float* const output,
                        const float* const input,
                        const size_t size,
                        const float value1,
                        const float value2,
                        const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    auto map_function = [value1, value2] __device__(const float input_pixel) -> float
    { return input_pixel == value1 || input_pixel == value2; };
    map_generic(output, input, size, map_function, stream);
}

void compute_first_mask_artery(float* const output,
                               const float* const input,
                               const size_t size,
                               const cudaStream_t stream)
{
    is_equal_to_either(output, input, size, 5, 4, stream);
}

void compute_first_mask_vein(float* const output,
                             const float* const input,
                             const size_t size,
                             const cudaStream_t stream)
{
    is_equal_to_either(output, input, size, 2, 3, stream);
}