#include "reduce.cuh"
#include "map.cuh"

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
kernel_multiply_frames_complex(cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        const float new_x = (input1[index].x * input2[index].x) - (input1[index].y * input2[index].y);
        const float new_y = (input1[index].y * input2[index].x) + (input1[index].x * input2[index].y);
        output[index].x = new_x;
        output[index].y = new_y;
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

void multiply_frames_complex(
    cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_frames_complex<<<blocks, threads, 0, stream>>>(output, input1, input2, size);
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

void tensor_multiply_vector(float* output,
                            const float* tensor,
                            const float* vector,
                            const size_t frame_res,
                            const ushort f_start,
                            const ushort f_end,
                            const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_tensor_multiply_vector<<<blocks, threads, 0, stream>>>(output, tensor, vector, frame_res, f_start, f_end);
    cudaCheckError();
}
