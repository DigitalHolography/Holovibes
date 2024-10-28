#pragma once
#include "convolution.cuh"
#include "fresnel_transform.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "common.cuh"
#include "cuda_memory.cuh"
#include <cufft.h>
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include <npp.h>

#include "matrix_operations.hh"
using holovibes::cuda_tools::CufftHandle;

void convolution_kernel(float* gpu_input,
                        float* gpu_convolved_buffer,
                        cuComplex* cuComplex_buffer,
                        CufftHandle* plan,
                        const size_t size,
                        const cuComplex* gpu_kernel,
                        const bool divide_convolution_enabled,
                        const bool normalize_enabled,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(size, threads);

    /* Copy gpu_input (float*) to cuComplex_buffer (cuComplex*)
     * We only want to copy the float value as real part float number in the
     * cuComplex_buffer To skip the imaginary part, we use a pitch (skipped
     * data) of size sizeof(float)
     *
     * The value are first all set to 0 (real & imaginary)
     * Then value are copied 1 by 1 from gpu_input into the real part
     * Imaginary is skipped and thus left to its value
     */
    cudaXMemsetAsync(cuComplex_buffer, 0, size * sizeof(cuComplex), stream);
    cudaSafeCall(cudaMemcpy2DAsync(cuComplex_buffer,  // Destination memory address
                                   sizeof(cuComplex), // Pitch of destination memory
                                   gpu_input,         // Source memory address
                                   sizeof(float),     // Pitch of source memory
                                   sizeof(float),     // Width of matrix transfer (columns in bytes)
                                   size,              // Height of matrix transfer (rows)
                                   cudaMemcpyDeviceToDevice,
                                   stream));
    // At this point, cuComplex_buffer is the same as the input

    cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_FORWARD));
    // At this point, cuComplex_buffer is the FFT of the input

    kernel_multiply_frames_complex<<<blocks, threads, 0, stream>>>(cuComplex_buffer,
                                                                   cuComplex_buffer,
                                                                   gpu_kernel,
                                                                   size);
    cudaCheckError();
    // At this point, cuComplex_buffer is the FFT of the input multiplied by the
    // FFT of the kernel

    cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_INVERSE));

    if (divide_convolution_enabled)
    {
        kernel_complex_to_modulus<<<blocks, threads, 0, stream>>>(cuComplex_buffer, gpu_convolved_buffer, size);
        cudaCheckError();
        kernel_divide_frames_float<<<blocks, threads, 0, stream>>>(gpu_input, gpu_convolved_buffer, gpu_input, size);
    }
    else
    {
        kernel_complex_to_modulus<<<blocks, threads, 0, stream>>>(cuComplex_buffer, gpu_input, size);
    }
    cudaCheckError();
}

__global__ void multiply_complex(cufftComplex* a, cufftComplex* b, cufftComplex* result, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        result[idx].x = a[idx].x * b[idx].x + a[idx].y * b[idx].y;
        result[idx].y = a[idx].y * b[idx].x - a[idx].x * b[idx].y;
    }
}

void xcorr2(float* output,
            float* input1,
            float* input2,
            const int freq_size,
            cudaStream_t stream,
            cufftComplex* d_freq_1,
            cufftComplex* d_freq_2,
            cufftComplex* d_corr_freq,
            cufftHandle plan_2d,
            cufftHandle plan_2dinv)
{
    cufftExecR2C(plan_2d, input1, d_freq_1);
    cufftExecR2C(plan_2d, input2, d_freq_2);

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(freq_size, threads);
    multiply_complex<<<blocks, threads, 0, stream>>>(d_freq_2, d_freq_1, d_corr_freq, freq_size);

    cufftExecC2R(plan_2dinv, d_corr_freq, output);
}