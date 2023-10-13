#include "cuda_memory.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "composite.cuh"

#include "logger.hh"

struct rect
{
    int x;
    int y;
    int w;
    int h;
};

struct comp
{
    ushort p_min;
    ushort p_max;
    float weight;
};

namespace
{
void check_zone(rect& zone, const uint frame_res, const int line_size)
{
    const int lines = line_size ? frame_res / line_size : 0;
    if (!zone.h || !zone.w || zone.x + zone.w > line_size || zone.y + zone.h > lines)
    {
        zone.x = 0;
        zone.y = 0;
        zone.w = line_size;
        zone.h = frame_res / line_size;
    }
}
} // namespace
__global__ static void
kernel_composite(cuComplex* input, float* output, const uint frame_res, size_t range, const float* colors)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        double res_components[3] = {0};
        for (ushort p = 0; p < range; p++)
        {
            cuComplex* current_pframe = input + (frame_res * p);
            float intensity = hypotf(current_pframe[id].x, current_pframe[id].y);
            for (int i = 0; i < 3; i++)
                res_components[i] += colors[p * 3 + i] * intensity;
        }
        for (int i = 0; i < 3; i++)
            output[id * 3 + i] = res_components[i] / range;
    }
}

// ! Splits the image by nb_lines blocks and sums them
__global__ static void kernel_sum_one_line(float* input,
                                           const uint frame_res,
                                           const uchar pixel_depth,
                                           const uint fd_line_size,
                                           const rect zone,
                                           float* sums_per_line)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < pixel_depth * zone.h)
    {
        uchar offset = id % pixel_depth; // offset e [0, 2] (RGB channel)
        ushort line = id / pixel_depth;  // line e [0, nb_line]
        line += zone.y;
        uint index_begin = fd_line_size * line + zone.x;
        uint index_end = index_begin + zone.w;
        if (index_end > frame_res)
            index_end = frame_res;
        float sum = 0;
        while (index_begin < index_end) // FIXME change to a real kernel reduce
            sum += input[pixel_depth * (index_begin++) + offset];
        sums_per_line[id] = sum;
    }
}

// ! sums an array of size floats and put the result divided by nb_elements in
// *output
__global__ static void
kernel_average_float_array(float* input, uint size, uint nb_elements, uint offset_per_pixel, float* output)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < offset_per_pixel)
    {
        input += id;
        float res = 0;
        while (size--)
        {
            res += *input;
            input += offset_per_pixel;
        }
        res /= static_cast<float>(nb_elements);
        output[id] = res;
    }
}

__global__ static void kernel_divide_by_weight(float* input, holovibes::RGBWeights weights)
{
    input[0] /= weights.r;
    input[1] /= weights.g;
    input[2] /= weights.b;
}
__global__ static void kernel_normalize_array(float* input, uint nb_pixels, uint pixel_depth, float* averages)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < pixel_depth * nb_pixels)
        input[id] = (input[id] / averages[id % 3]) * 1000;
    // The * 1000 is used to have the result in [0;1000]
    // instead of [0;1] for a better contrast control
}

__global__ static void
kernel_precompute_colors(float* colors, size_t red, size_t blue, size_t range, holovibes::RGBWeights weights)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < range)
    {
        double hue = double(id) / double(range); // hue e [0,1]
        hue *= double(range) / double(range-1);
        if (hue < 0.25)
        {
            colors[id * 3 + 0] = weights.r;
            colors[id * 3 + 1] = (hue / 0.25) * weights.g;
            colors[id * 3 + 2] = 0;
        }
        else if (hue < 0.5)
        {
            colors[id * 3 + 0] = (1 - (hue - 0.25) / 0.25) * weights.r;
            colors[id * 3 + 1] = weights.g;
            colors[id * 3 + 2] = 0;
        }
        else if (hue < 0.75)
        {
            colors[id * 3 + 0] = 0;
            colors[id * 3 + 1] = weights.g;
            colors[id * 3 + 2] = ((hue - 0.5) / 0.25) * weights.b;
        }
        else
        {
            colors[id * 3 + 0] = 0;
            colors[id * 3 + 1] = (1 - (hue - 0.75) / 0.25) * weights.g;
            colors[id * 3 + 2] = weights.b;
        }
    }
}

void rgb(cuComplex* input,
         float* output,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         holovibes::RGBWeights weights,
         const cudaStream_t stream)
{
    ushort range = std::abs(static_cast<short>(max - min)) + 1;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(range, threads);

    size_t colors_size = range * 3;
    holovibes::cuda_tools::UniquePtr<float> colors(colors_size);

    if (auto_weights)
    {
        holovibes::RGBWeights raw_color = {0};
        raw_color.r = 1;
        raw_color.g = 1;
        raw_color.b = 1; // (1,1,1) = raw colors
        kernel_precompute_colors<<<blocks, threads, 0, stream>>>(colors.get(), min, max, range, raw_color);
    }
    else
        kernel_precompute_colors<<<blocks, threads, 0, stream>>>(colors.get(), min, max, range, weights);

    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_composite<<<blocks, threads, 0, stream>>>(input, output, frame_res, range, colors.get());

    cudaCheckError();
}

void postcolor_normalize(float* output,
                         const size_t frame_res,
                         const uint fd_line_size,
                         holovibes::units::RectFd selection,
                         const uchar pixel_depth,
                         holovibes::RGBWeights weights,
                         float* averages,
                         const cudaStream_t stream)
{
    rect zone = {selection.x(), selection.y(), selection.unsigned_width(), selection.unsigned_height()};
    check_zone(zone, frame_res, fd_line_size);
    float* gpu_averages = nullptr;
    float* sums_per_line = nullptr;
    cudaXMalloc(&gpu_averages, sizeof(float) * pixel_depth);
    cudaXMalloc(&sums_per_line, sizeof(float) * zone.h * pixel_depth);

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(zone.h * pixel_depth, threads);

    kernel_sum_one_line<<<blocks, threads, 0, stream>>>(output,
                                                        frame_res,
                                                        pixel_depth,
                                                        fd_line_size,
                                                        zone,
                                                        sums_per_line);
    cudaCheckError();

    blocks = map_blocks_to_problem(pixel_depth, threads);
    kernel_average_float_array<<<blocks, threads, 0, stream>>>(sums_per_line,
                                                               zone.h,
                                                               zone.h * zone.w,
                                                               pixel_depth,
                                                               gpu_averages);
    cudaCheckError();

    // kernel_divide_by_weight<<<1, 1, 0, stream>>>(averages, weight_r, weight_g, weight_b); // Using a kernel for this
    // simple operation may preserve the stream pipeline cudaCheckError();

    cudaXMemcpyAsync(averages, gpu_averages, sizeof(float) * pixel_depth, cudaMemcpyDeviceToHost, stream);

    blocks = map_blocks_to_problem(frame_res * pixel_depth, threads);
    kernel_normalize_array<<<blocks, threads, 0, stream>>>(output, frame_res, pixel_depth, gpu_averages);
    cudaCheckError();

    cudaXFree(gpu_averages);
    cudaXFree(sums_per_line);
}
