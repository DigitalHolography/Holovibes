#include "cuda_memory.cuh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include "tools_analysis_debug.hh"

__global__ void kernel_minus_negation_times_2(float* R_vascular_pulse, float* mask_vesselnessClean, uint size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        R_vascular_pulse[idx] = R_vascular_pulse[idx] - !mask_vesselnessClean[idx] * 2;
}

void minus_negation_times_2(float* R_vascular_pulse, float* mask_vesselnessClean, uint size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_minus_negation_times_2<<<blocks, threads, 0, stream>>>(R_vascular_pulse, mask_vesselnessClean, size);
    cudaCheckError();
}

__global__ void kernel_negation(float* input_output, uint size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        input_output[idx] = !input_output[idx];
}

void negation(float* input_output, uint size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_negation<<<blocks, threads, 0, stream>>>(input_output, size);
    cudaCheckError();
}

__global__ void kernel_quantize(float* output, float* input, float* thresholds, int length_input, int lenght_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Si l'index est dans la plage du tableau d'entrée
    if (idx < length_input)
    {
        float value = input[idx];
        int quantized_level = 1;

        // Trouver le niveau de quantification en fonction des seuils
        for (int t = 0; t < lenght_threshold; ++t)
        {
            if (value > thresholds[t])
                quantized_level = t + 2;
            else
                break;
        }

        // Stocker le résultat
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

__global__ void kernel_is_both_value(float* output, float* input, uint size, float value1, float value2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        output[idx] = input[idx] == value1 || input[idx] == value2;
}

void is_both_value(float* output, float* input, uint size, float value1, float value2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_is_both_value<<<blocks, threads, 0, stream>>>(output, input, size, value1, value2);
    cudaCheckError();
}

void compute_first_mask_artery(float* output, float* input, uint size, cudaStream_t stream)
{
    is_both_value(output, input, size, 5, 4, stream);
}

void compute_first_mask_vein(float* output, float* input, uint size, cudaStream_t stream)
{
    is_both_value(output, input, size, 2, 3, stream);
}