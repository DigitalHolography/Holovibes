#include "cuda_memory.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "rgb.cuh"
#include "map.cuh"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "logger.hh"

/**
 * @brief Apply the colors on the image cube
 * @param input input complex buffer
 * @param output output rgb float buffer
 * @param frame_res The total number of pixel in one frame
 * @param range The number of frame on depth (max - min)
 * @param colors The computed color buffer
 * @return
 */
__global__ static void
kernel_composite(cuComplex* input, float* output, const uint frame_res, size_t range, const float* colors)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float res_components[3] = {0};
        for (ushort p = 0; p < range; p++)
        {
            cuComplex* current_pframe = input + (frame_res * p);
            float intensity = hypotf(current_pframe[id].x, current_pframe[id].y);
            for (int i = 0; i < 3; i++)
                res_components[i] = __fmaf_rn(colors[p * 3 + i], intensity, res_components[i]);
        }
        for (int i = 0; i < 3; i++)
            output[id * 3 + i] = __fdiv_rn(res_components[i], range);
    }
}

/**
 * @brief Compute the actual color of the pixel based on the depth of the frame, equivalent to sampling the rgb color
 * gradient on a position
 * @param colors The buffer to fill (range * 3 * sizeof(float))
 * @param range The number of frame on depth (max - min)
 * @param weights The weights entered on the UI
 */
__global__ static void kernel_precompute_colors(float* colors, size_t range, holovibes::RGBWeights weights)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < range)
    {
        double hue = double(id) / double(range); // hue e [0,1]
        hue *= double(range) / double(range - 1);
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

/**
 * @brief Compute the rgb color of each pixel of the image
 * @param input The input complex buffer
 * @param output The output rgb float buffer (1 pixel = 3 floats)
 * @param frame_res The total number of pixels in an image
 * @param auto_weights A boolean equal to the value of the auto equalization checkbox
 * @param min Starting depth in image cube
 * @param max Last frame index in image cube
 * @param weights The RGB weights
 * @param stream The cuda stream used
 */
void rgb(cuComplex* input,
         float* output,
         const size_t frame_res,
         bool auto_weights,
         const ushort min,
         const ushort max,
         holovibes::RGBWeights weights,
         const cudaStream_t stream)
{
    input = input + (min * frame_res);
    ushort range = std::abs(static_cast<short>(max - min)) + 1;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(range, threads);

    size_t colors_size = range * 3;
    holovibes::cuda_tools::CudaUniquePtr<float> colors(colors_size);

    if (auto_weights)
    {
        holovibes::RGBWeights raw_color = {0};
        raw_color.r = 1;
        raw_color.g = 1;
        raw_color.b = 1; // (1,1,1) = raw colors
        kernel_precompute_colors<<<blocks, threads, 0, stream>>>(colors.get(), range, raw_color);
    }
    else
        kernel_precompute_colors<<<blocks, threads, 0, stream>>>(colors.get(), range, weights);

    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_composite<<<blocks, threads, 0, stream>>>(input, output, frame_res, range, colors.get());
    // composite_block(input, output, frame_res, colors.get(), range, stream);

    cudaCheckError();
}