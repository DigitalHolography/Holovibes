#pragma once
#include "convolution.cuh"
#include "fresnel_transform.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "common.cuh"
#include "cuda_memory.cuh"

#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"

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

void convolution_float(float* output,
                       const float* input1,
                       const float* input2,
                       const uint size,
                       const cufftHandle plan2d_a,
                       const cufftHandle plan2d_b,
                       const cufftHandle plan2d_inverse,
                       cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    // The convolution operator could be optimized.
    // TODO: pre allocate tmp buffers and pass them to the function
    holovibes::cuda_tools::CudaUniquePtr<cuComplex> tmp_a(size);
    holovibes::cuda_tools::CudaUniquePtr<cuComplex> tmp_b(size);
    if (!tmp_a || !tmp_b)
        return;

    cufftExecR2C(plan2d_a, const_cast<float*>(input1), tmp_a.get());
    cufftExecR2C(plan2d_b, const_cast<float*>(input2), tmp_b.get());

    cudaStreamSynchronize(0);

    kernel_multiply_frames_complex<<<blocks, threads, 0, stream>>>(tmp_a.get(), tmp_a.get(), tmp_b.get(), size);
    cudaCheckError();
    cudaStreamSynchronize(stream);
    cufftExecC2R(plan2d_inverse, tmp_a.get(), output);
    cudaStreamSynchronize(0);
}

__global__ void cross_correlation_2d(const float* input, const float* kernel, float* output, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float result = 0.0f;

        for (int ky = 0; ky < height; ++ky)
        {
            for (int kx = 0; kx < width; ++kx)
            {
                int ix = x + kx - width / 2;
                int iy = y + ky - height / 2;

                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                {
                    result += input[iy * width + ix] * kernel[ky * width + kx];
                }
            }
        }

        output[y * width + x] = result;
    }
}

void xcorr2(
    float* output, const float* input1, const float* input2, const short width, const short height, cudaStream_t stream)
{

    // CufftHandle plan2d_1(height, width, CUFFT_R2C);
    // CufftHandle plan2d_2(height, width, CUFFT_R2C);
    // CufftHandle plan2d_inverse(height, width, CUFFT_C2R);

    // convolution_float(output, input1, input2, width * height, plan2d_1, plan2d_2, plan2d_inverse, stream);

    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);
    cross_correlation_2d<<<lblocks, lthreads, 0, stream>>>(input1, input2, output, width, height);
    cudaCheckError();
}