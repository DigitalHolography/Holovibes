#include "delete_twin_image.cuh"

namespace
{
__global__ void gaussian_blur_horizontal_kernel(
    const float* input, float* output, const float* kernel, int radius, int width, int height, size_t frame_stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int frame = blockIdx.z;

    if (x >= width || y >= height)
        return;

    const size_t base = static_cast<size_t>(frame) * frame_stride + static_cast<size_t>(y) * width;

    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k)
    {
        int xx = x + k;
        if (xx < 0)
            xx = 0;
        else if (xx >= width)
            xx = width - 1;

        acc += input[base + xx] * kernel[k + radius];
    }

    output[base + x] = acc;
}

__global__ void gaussian_blur_vertical_kernel(
    const float* input, float* output, const float* kernel, int radius, int width, int height, size_t frame_stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int frame = blockIdx.z;

    if (x >= width || y >= height)
        return;

    const size_t frame_offset = static_cast<size_t>(frame) * frame_stride;
    float acc = 0.0f;
    for (int k = -radius; k <= radius; ++k)
    {
        int yy = y + k;
        if (yy < 0)
            yy = 0;
        else if (yy >= height)
            yy = height - 1;

        acc += input[frame_offset + static_cast<size_t>(yy) * width + x] * kernel[k + radius];
    }

    output[frame_offset + static_cast<size_t>(y) * width + x] = acc;
}

__global__ void
subtract_arrays_kernel(float* output, const float* minuend, const float* subtrahend, size_t element_count)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count)
        output[index] = minuend[index] - subtrahend[index];
}

__global__ void combine_amplitude_phase_kernel(
    cuComplex* output, const float* amplitude, const float* phase, size_t element_count, bool negate_phase)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count)
        return;

    const float phase_value = negate_phase ? -phase[index] : phase[index];
    float s;
    float c;
    s = sinf(phase_value);
    c = cosf(phase_value);
    const float amp = amplitude[index];
    output[index].x = amp * c;
    output[index].y = amp * s;
}
} // namespace

void gaussian_blur_batch(const float* input,
                         float* temp,
                         float* output,
                         const float* kernel,
                         int radius,
                         int width,
                         int height,
                         uint batch_size,
                         const cudaStream_t stream)
{
    if (!input || !temp || !output || !kernel || batch_size == 0 || width <= 0 || height <= 0 || radius <= 0)
        return;

    const size_t frame_stride = static_cast<size_t>(width) * height;
    const dim3 block(16, 16, 1);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);

    gaussian_blur_horizontal_kernel<<<grid, block, 0, stream>>>(input,
                                                                temp,
                                                                kernel,
                                                                radius,
                                                                width,
                                                                height,
                                                                frame_stride);
    gaussian_blur_vertical_kernel<<<grid, block, 0, stream>>>(temp,
                                                              output,
                                                              kernel,
                                                              radius,
                                                              width,
                                                              height,
                                                              frame_stride);
    cudaCheckError();
}

void subtract_arrays(
    float* output, const float* minuend, const float* subtrahend, size_t element_count, const cudaStream_t stream)
{
    if (!output || !minuend || !subtrahend || element_count == 0)
        return;

    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(element_count, threads);
    subtract_arrays_kernel<<<blocks, threads, 0, stream>>>(output, minuend, subtrahend, element_count);
    cudaCheckError();
}

void combine_amplitude_phase(cuComplex* output,
                             const float* amplitude,
                             const float* phase,
                             size_t element_count,
                             bool negate_phase,
                             const cudaStream_t stream)
{
    if (!output || !amplitude || !phase || element_count == 0)
        return;

    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(element_count, threads);
    combine_amplitude_phase_kernel<<<blocks, threads, 0, stream>>>(output,
                                                                   amplitude,
                                                                   phase,
                                                                   element_count,
                                                                   negate_phase);
    cudaCheckError();
}
