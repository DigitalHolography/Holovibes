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
    for (uint i = f_start; i <= f_end; i++)
    {
        const float* current_frame = tensor + i * frame_res;
        val += current_frame[index] * vector[i];
    }

    // Nyquist frequency handling, when even time window must be doubled for M0/M2, zero for M1
    // TODO: check if nyquist frequency should be counted double in other places of the code
    if (even && nyquist_index >= f_start && nyquist_index <= f_end)
    {
        const float* current_frame = tensor + nyquist_index * frame_res;
        if (m1)
            val -= current_frame[index] * vector[nyquist_index];
        else
            val += current_frame[index] * vector[nyquist_index];
    }

    output[index] = val;
}

void tensor_multiply_vector_nyquist_compensation(float* output,
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

/*!
 * \brief Computes the element-wise Hadamard product of two input arrays.
 *
 * This CUDA kernel function computes the element-wise Hadamard product (element-wise multiplication) of two input
 * arrays, `input1` and `input2`. The result is stored in the output array. Each thread in the kernel computes the
 * product for a single element.
 *
 * \param [out] output Pointer to the output array where the result of the Hadamard product will be stored.
 * \param [in] input1 Pointer to the first input array.
 * \param [in] input2 Pointer to the second input array.
 * \param [in] size The number of elements in the input arrays.
 *
 * \note The function performs the Hadamard product only for the elements within the specified size.
 */
__global__ void kernel_compute_hadamard_product(float* const output,
                                                const float* const input1,
                                                const float* const input2,
                                                const size_t size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input1[index] * input2[index];
}

void compute_hadamard_product(
    float* output, const float* const input1, const float* const input2, const size_t size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_compute_hadamard_product<<<blocks, threads, 0, stream>>>(output, input1, input2, size);
    cudaCheckError();
}
