#include "cuda_memory.cuh"
#include "map.cuh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

__global__ void kernel_minus_negation_times_2(float* R_vascular_pulse, float* mask_vesselnessClean, uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        R_vascular_pulse[index] -= !mask_vesselnessClean[index] * 2;
}

void minus_negation_times_2(float* R_vascular_pulse, float* mask_vesselnessClean, uint size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_minus_negation_times_2<<<blocks, threads, 0, stream>>>(R_vascular_pulse, mask_vesselnessClean, size);
    cudaCheckError();
}

void negation(float* input_output, uint size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    auto map_function = [] __device__(const float input_pixel) -> float { return !input_pixel; };
    map_generic(input_output, size, map_function, stream);
}

__global__ void kernel_quantize(float* output, float* input, float* thresholds, int length_input, int lenght_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length_input)
    {
        float value = input[idx];
        int quantized_level = 1;

        for (int t = 0; t < lenght_threshold; ++t)
        {
            if (value > thresholds[t])
                quantized_level = t + 2;
            else
                break;
        }

        output[idx] = quantized_level;
    }
}

void imquantize(
    float* output, float* input, float* thresholds, int length_input, int lenght_threshold, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(length_input, threads);
    kernel_quantize<<<blocks, threads, 0, stream>>>(output, input, thresholds, length_input, lenght_threshold);
    cudaCheckError();
}

void segment_vessels(float* output,
                     float* new_thresholds,
                     float* R_VascularPulse,
                     float* mask_vesselness_clean,
                     uint size,
                     float* thresholds,
                     cudaStream_t stream)
{
    float minus_one = -1;
    cudaXMemcpyAsync(new_thresholds + 1, thresholds, sizeof(float) * 3, cudaMemcpyHostToDevice, stream);
    cudaXMemcpyAsync(new_thresholds, &minus_one, sizeof(float), cudaMemcpyHostToDevice, stream);

    minus_negation_times_2(R_VascularPulse, mask_vesselness_clean, size, stream);
    imquantize(output, R_VascularPulse, new_thresholds, size, 4, stream);
}

void is_equal_to_either(float* output, float* input, uint size, float value1, float value2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    auto map_function = [value1, value2] __device__(const float input_pixel) -> float
    { return input_pixel == value1 || input_pixel == value2; };
    map_generic(output, input, size, map_function, stream);
}

void compute_first_mask_artery(float* output, float* input, uint size, cudaStream_t stream)
{
    is_equal_to_either(output, input, size, 5, 4, stream);
}

void compute_first_mask_vein(float* output, float* input, uint size, cudaStream_t stream)
{
    is_equal_to_either(output, input, size, 2, 3, stream);
}