#include "reduce.cuh"
#include "map.cuh"
#include "tools_compute.cuh"
#include "tools.cuh"

#include <stdio.h>

#define AUTO_CONTRAST_COMPENSATOR 10000

__global__ void
kernel_complex_divide(cuComplex* image, const uint frame_res, const float divider, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < frame_res)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            const uint batch_index = index + i * frame_res;

            image[batch_index].x /= divider;
            image[batch_index].y /= divider;
        }
    }
}

__global__ void
kernel_divide_frames_float(float* output, const float* numerator, const float* denominator, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = numerator[index] / denominator[index];
        output[index] = new_x;
    }
}

__global__ void kernel_tensor_multiply_vector_nyquist_compensation(float* output,
                                                                   const float* tensor,
                                                                   const float* vector,
                                                                   const size_t frame_res,
                                                                   const ushort f_start,
                                                                   const ushort f_end,
                                                                   const size_t nyquist_index,
                                                                   const bool even,
                                                                   const bool m1)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= frame_res)
        return;

    float val = 0.0f;
    uint i = f_start;
    for (; i < nyquist_index; i++)
    {
        const float* current_frame = tensor + i * frame_res;
        val += current_frame[index] * vector[i];
    }

    // Nyquist frequency handling, when even time window must be doubled for M0/M2, zero for M1
    // Boolean arithmetic logic is used to avoid conditional in kernel
    const float* current_frame = tensor + nyquist_index * frame_res;
    val += (i <= f_end) * (1.f + even) * current_frame[index] * vector[nyquist_index] * !(m1 && even);

    for (i = nyquist_index + 1; i <= f_end; i++)
    {
        const float* current_frame = tensor + i * frame_res;
        val += current_frame[index] * vector[i];
    }

    output[index] = val;
}

__global__ void kernel_tensor_multiply_vector(float* output,
                                              const float* tensor,
                                              const float* vector,
                                              const size_t frame_res,
                                              const ushort f_start,
                                              const ushort f_end)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= frame_res)
        return;

    float val = 0.0f;
    for (uint i = f_start; i <= f_end; i++)
    {
        const float* current_frame = tensor + i * frame_res;
        val += current_frame[index] * vector[i];
    }

    output[index] = val;
}

void tensor_multiply_vector(float* output,
                            const float* tensor,
                            const float* vector,
                            const size_t frame_res,
                            const ushort f_start,
                            const ushort f_end,
                            const size_t nyquist_freq,
                            bool even,
                            bool m1,
                            const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_tensor_multiply_vector_nyquist_compensation<<<blocks, threads, 0, stream>>>(output,
                                                                                       tensor,
                                                                                       vector,
                                                                                       frame_res,
                                                                                       f_start,
                                                                                       f_end,
                                                                                       nyquist_freq,
                                                                                       even,
                                                                                       m1);
    cudaCheckError();
}

void gpu_normalize(float* const input,
                   double* const result_reduce,
                   const size_t frame_res,
                   const uint norm_constant,
                   const cudaStream_t stream)
{
    reduce_add(result_reduce, input, frame_res, stream);

    /* Let x be a pixel, after renormalization
    ** x = x * 2^(norm_constant) / mean
    ** x = x * 2^(norm_constant) * frame_res / reduce_result
    ** x = x * 2^(norm_constant) * (frame_res / reduce_result)
    */
    const float multiplier = (1 << norm_constant);
    auto map_function = [multiplier, frame_res, result_reduce] __device__(const float input_pixel) -> float
    {
        /* Computing on double is really slow on a GPU, in our case
         *result_reduce can never overflow
         ** Thus it can be casted to a float
         */
        return input_pixel * multiplier * (frame_res / static_cast<const float>(*result_reduce));
    };

    map_generic(input, input, frame_res, map_function, stream);
}
