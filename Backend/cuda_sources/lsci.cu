#include "lsci.cuh"

#include <stdexcept>

namespace
{
template <typename T>
__global__ void accumulate_intensity(float2* moments, const T* input, size_t frame_res, unsigned batch_size, bool reset)
{
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= frame_res)
        return;

    float2 value = reset ? make_float2(0.f, 0.f) : moments[pixel];
    for (unsigned frame = 0; frame < batch_size; ++frame)
    {
        const float intensity = static_cast<float>(input[frame * frame_res + pixel]);
        value.x += intensity;
        value.y += intensity * intensity;
    }
    moments[pixel] = value;
}

__global__ void pooled_contrast(float* output,
                                const float2* moments,
                                unsigned width,
                                unsigned height,
                                unsigned frame_count,
                                unsigned spatial_window)
{
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= static_cast<size_t>(width) * height)
        return;

    const int x = static_cast<int>(pixel % width);
    const int y = static_cast<int>(pixel / width);
    const int radius = static_cast<int>(spatial_window / 2);
    const int first_x = max(0, x - radius);
    const int last_x = min(static_cast<int>(width) - 1, x + radius);
    const int first_y = max(0, y - radius);
    const int last_y = min(static_cast<int>(height) - 1, y + radius);

    float sum = 0.f;
    float squared_sum = 0.f;
    for (int row = first_y; row <= last_y; ++row)
    {
        for (int column = first_x; column <= last_x; ++column)
        {
            const float2 value = moments[static_cast<size_t>(row) * width + column];
            sum += value.x;
            squared_sum += value.y;
        }
    }

    const float count = static_cast<float>(last_x - first_x + 1) * (last_y - first_y + 1) * frame_count;
    const float mean = sum / count;
    const float variance = fmaxf(squared_sum / count - mean * mean, 0.f);
    output[pixel] = mean > 0.f ? sqrtf(variance) / mean : 0.f;
}
} // namespace

void lsci_accumulate(float2* moments,
                     const void* input,
                     size_t frame_res,
                     unsigned batch_size,
                     camera::PixelDepth depth,
                     bool reset,
                     cudaStream_t stream)
{
    if (frame_res == 0 || batch_size == 0)
        return;

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((frame_res + threads - 1) / threads);
    switch (depth)
    {
    case camera::PixelDepth::Bits8:
        accumulate_intensity<<<blocks, threads, 0, stream>>>(moments,
                                                             static_cast<const unsigned char*>(input),
                                                             frame_res,
                                                             batch_size,
                                                             reset);
        break;
    case camera::PixelDepth::Bits16:
        accumulate_intensity<<<blocks, threads, 0, stream>>>(moments,
                                                             static_cast<const unsigned short*>(input),
                                                             frame_res,
                                                             batch_size,
                                                             reset);
        break;
    case camera::PixelDepth::Bits32:
        accumulate_intensity<<<blocks, threads, 0, stream>>>(moments,
                                                             static_cast<const float*>(input),
                                                             frame_res,
                                                             batch_size,
                                                             reset);
        break;
    default:
        throw std::invalid_argument("LSCI requires 8-bit, 16-bit, or float intensity frames");
    }
    cudaCheckError();
}

void lsci_contrast(float* output,
                   const float2* moments,
                   unsigned width,
                   unsigned height,
                   unsigned frame_count,
                   cudaStream_t stream,
                   unsigned spatial_window)
{
    if (spatial_window > 63 || spatial_window % 2 == 0)
        throw std::invalid_argument("LSCI spatial window must be an odd side length from 1 through 63");

    if (width == 0 || height == 0 || frame_count == 0)
        return;

    constexpr unsigned threads = 256;
    const size_t frame_res = static_cast<size_t>(width) * height;
    const unsigned blocks = static_cast<unsigned>((frame_res + threads - 1) / threads);
    pooled_contrast<<<blocks, threads, 0, stream>>>(output, moments, width, height, frame_count, spatial_window);
    cudaCheckError();
}
