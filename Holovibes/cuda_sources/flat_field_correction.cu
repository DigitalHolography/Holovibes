#include "unique_ptr.hh"
#include "common.cuh"
#include "cuda_memory.cuh"
#include "reduce.cuh"
#include "convolution.cuh"
#include <algorithm>

__global__ void kernel_normalize_image(float* const output, const float* const input, const size_t height, const size_t width, const float im_min, const float im_diff)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < height * width)
        output[index] = (input[index] - im_min) / im_diff;
}

__global__ void kernel_flat_field_correction(float* const output, const loat* const input, const size_t height, const size_t width)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < height * width)
    {}
}

void apply_flat_field_correction(float* const output, const float* const input, const size_t height, const size_t width, const float gw, const float border_amount, const cudaStream_t stream)
{
    const size_t size = width * height;
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    // Check if the input needs to be normalized between 0 and 1
    int flag = 0;
    // TODO: use reduce instead
    const float im_max = std::max_element(input, size);
    const float im_min = std::min_element(input, size);
    const float im_diff = im_max - im_min;
    
    if (flag = (im_max > 1.0f || im_min < 0.0f))
        kernel_normalize_image<<<blocks, threads, 0, stream>>>(output, input, height, width, im_min, im_diff);
    else
        cudaXMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    int a = 1;
    int b = height;
    int c = 1;
    int d = width;
    if (border_amount != 0.0f)
    {
        a = std::ceil(height * border_amount);
        b = std::floor(height * (1 - border_amount));
        c = std::ceil(width * border_amount);
        d = std::floor(width * (1 - border_amount));
    }
    
    // TODO : use reduce instead
    float sum = 0.0f;
    for (size_t i = a; i <= b; ++i)
    {
        for (size_t j = c; j <= d; ++j)
        {
            sum += output[width * i + j];
        }
    }

    
    // Apply a Gaussian filter
    float* convolution_buffer;
    cuComplex* complex_buffer;
    cuComplex* kernel_buffer;

    cudaXMalloc(&convolution_buffer, size * sizeof(float));
    cudaXMalloc(&complex_buffer, size * sizeof(cuComplex));
    cudaXMalloc(&kernel_buffer, size * sizeof(cuComplex));

    float gauss_kernel[5][5] = {
    {0.00398, 0.01517, 0.02384, 0.01517, 0.00398},
    {0.01517, 0.05855, 0.09158, 0.05855, 0.01517},
    {0.02384, 0.09158, 0.14600, 0.09158, 0.02384},
    {0.01517, 0.05855, 0.09158, 0.05855, 0.01517},
    {0.00398, 0.01517, 0.02384, 0.01517, 0.00398},
    };


    cudaXMemsetAsync(kernel_buffer, 0, size * sizeof(cuComplex), stream);
    cudaMemcpy2DAsync(kernel_buffer,
                        sizeof(cuComplex),
                        gauss_kernel,
                        5 * sizeof(float),
                        5 * sizeof(float),
                        5,
                        cudaMemcpyHostToDevice,
                        stream);

    convolution_kernel(output,
                       convolution_buffer,
                       complex_buffer,
                       &convolution_plan_,
                       width,
                       kernel_buffer,
                       false,
                       stream);
}