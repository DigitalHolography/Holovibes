#include "vascular_pulse.cuh"

#include "cuda_memory.cuh"
#include "tools_analysis_debug.hh"
#include "compute_env.hh"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cmath>

__global__ void kernel_divide_constant(float* vascular_pulse, int value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] /= value;
}

void divide_constant(float* vascular_pulse, int value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
    cudaCheckError();
}

__global__ void kernel_divide(float* input_output, float* denominator_array, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        input_output[index] /= denominator_array[index];
}

void divide(float* input_output, float* denominator_array, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_divide<<<blocks, threads, 0, stream>>>(input_output, denominator_array, size);
    cudaCheckError();
}

__global__ void kernel_multiply_constant(float* vascular_pulse, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        vascular_pulse[index] *= value;
}

void multiply_constant(float* vascular_pulse, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_constant<<<blocks, threads, 0, stream>>>(vascular_pulse, value, size);
    cudaCheckError();
}

float compute_mean(float* vascular_pulse, size_t size)
{
    thrust::device_ptr<float> d_array(vascular_pulse);

    float sum = thrust::reduce(d_array, d_array + size, 0.0f, thrust::plus<float>());

    return sum / size;
}

__global__ void kernel_subtract_constant(float* output, float* input, float value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input[index] - value;
}

void subtract_constant(float* output, float* input, float value, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_subtract_constant<<<blocks, threads, 0, stream>>>(output, input, value, size);
    cudaCheckError();
}

__global__ void kernel_multiply_three_vectors(float* output, float* input1, float* input2, float* input3, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input1[index] * input2[index] * input3[index];
}

void multiply_three_vectors(
    float* output, float* input1, float* input2, float* input3, size_t size, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_three_vectors<<<blocks, threads, 0, stream>>>(output, input1, input2, input3, size);
    cudaCheckError();
}

// __global__ void
// kernel_computeMean(const float* M0, const float* vascularPulse, float* result, int rows, int cols, int depth)
// {
//     // Calcul des indices globaux
//     int x = blockIdx.x * blockDim.x + threadIdx.x; // index de ligne
//     int y = blockIdx.y * blockDim.y + threadIdx.y; // index de colonne

//     if (x < rows && y < cols)
//     {
//         float sum = 0.0f;

//         // Somme sur la 3ème dimension
//         for (int z = 0; z < depth; ++z)
//         {
//             int index3D = x * cols + y + z * rows * cols;
//             sum += M0[index3D] * vascularPulse[z];
//         }

//         // Stocker la moyenne dans le tableau résultat
//         result[x * cols + y] = sum / depth;
//     }
// }

__global__ void kernel_computeMean(const float* M0_ff_video_centered,
                                   const float* vascularPulse_centered,
                                   float* result,
                                   int rows,
                                   int cols,
                                   int depth)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < rows * cols)
    {
        float sum = 0.0f;
        for (int z = 0; z < depth; z++)
        {
            int idx_video = z * cols * rows + index;
            int idx_pulse = z; // puisque vascularPulse_centered est 1x1xDEPTH
            sum += M0_ff_video_centered[idx_video] * vascularPulse_centered[idx_pulse];
        }
        result[index] = sum / depth;
    }
}

void computeMean(
    const float* M0, const float* vascularPulse, float* result, int rows, int cols, int depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(rows * cols, threads);
    // dim3 blockSize(16, 16);
    // dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    kernel_computeMean<<<blocks, threads, 0, stream>>>(M0, vascularPulse, result, rows, cols, depth);
    cudaCheckError();
}

__global__ void kernel_compute_std(const float* input, float* output, int size, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        double mean = 0.0f;
        double variance = 0.0f;

        // Compute mean along the third dimension
        for (int k = 0; k < depth; ++k)
        {
            mean += input[idx + size * k];
        }
        mean /= depth - 1;

        // Compute variance along the third dimension
        for (int k = 0; k < depth; ++k)
        {
            float diff = input[idx + size * k] - mean;
            variance += diff * diff;
        }
        variance /= depth - 1;

        // Store the standard deviation in the output array
        output[idx] = (float)sqrt(variance);
    }
}

void compute_std(const float* input, float* output, int size, int depth, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_compute_std<<<blocks, threads, 0, stream>>>(input, output, size, depth);
    cudaCheckError();
}

void compute_first_correlation(float* output,
                               float* M0_ff_video_centered,
                               float* vascular_pulse,
                               int nnz_mask_vesslness_clean,
                               size_t length_video,
                               VesselnessFilterStruct& filter_struct_,
                               size_t image_size,
                               cudaStream_t stream) // Size here is future time window
{
    divide_constant(vascular_pulse, nnz_mask_vesslness_clean, length_video, stream);

    float vascular_mean = compute_mean(vascular_pulse, length_video);
    subtract_constant(filter_struct_.vascular_pulse_centered, vascular_pulse, vascular_mean, length_video, stream);

    computeMean(M0_ff_video_centered, filter_struct_.vascular_pulse_centered, output, 512, 512, length_video, stream);

    compute_std(M0_ff_video_centered, filter_struct_.std_M0_ff_video_centered, 512 * 512, length_video, stream);

    compute_std(filter_struct_.vascular_pulse_centered,
                filter_struct_.std_vascular_pulse_centered,
                1,
                length_video,
                stream);

    float std_vascular_pulse_centered_cpu;
    cudaXMemcpy(&std_vascular_pulse_centered_cpu,
                filter_struct_.std_vascular_pulse_centered,
                sizeof(float),
                cudaMemcpyDeviceToHost);

    multiply_constant(filter_struct_.std_M0_ff_video_centered, std_vascular_pulse_centered_cpu, 512 * 512, stream);

    divide(output, filter_struct_.std_M0_ff_video_centered, 512 * 512, stream);
}