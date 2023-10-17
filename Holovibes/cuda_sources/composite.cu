#include "cuda_memory.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "composite.cuh"
#include "map.cuh"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "logger.hh"

/**
 * @brief A struct to represent a RGB pixel.
 */
struct RGBPixel
{
    float r;
    float g;
    float b;

    RGBPixel __host__ __device__ operator+(const RGBPixel& other) const
    {
        RGBPixel result;
        result.r = this->r + other.r;
        result.g = this->g + other.g;
        result.b = this->b + other.b;
        return result;
    }

    RGBPixel __host__ __device__ operator/(float scalar) const
    {
        RGBPixel result;
        result.r = this->r / scalar;
        result.g = this->g / scalar;
        result.b = this->b / scalar;
        return result;
    }

    RGBPixel __host__ __device__ operator*(float scalar) const
    {
        RGBPixel result;
        result.r = this->r * scalar;
        result.g = this->g * scalar;
        result.b = this->b * scalar;
        return result;
    }
};

struct rect
{
    int x;
    int y;
    int w;
    int h;
};

/**
 * @brief Check that the selected zone is not out of bounds, if so replace it to take the entiere image 
 * @param zone The selected zone on the UI
 * @param frame_res The total number of pixel in one frame
 * @param line_size The length of one frame line
 */
static void check_zone(rect& zone, const uint frame_res, const int line_size)
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
 * @brief Compute the actual color of the pixel based on the depth of the frame, equivalent to sampling the rgb color gradient on a position
 * @param colors The buffer to fill (range * 3 * sizeof(float))
 * @param range The number of frame on depth (max - min)
 * @param weights The weights entered on the UI
 */
__global__ static void
kernel_precompute_colors(float* colors, size_t range, holovibes::RGBWeights weights)
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
    holovibes::cuda_tools::UniquePtr<float> colors(colors_size);

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

/**
 * @brief Copy the selected zone to a contiguous rgb pixel buffer
 * @param input The whole rgb image
 * @param zone_data The output rgb buffer
 * @param zone The selected zone characteristics
 * @param range The total number of pixel to copy
 * @param fd_width The width of the input image
 * @param start_zone_id The index of the beginning of the input buffer
 */
__global__ static void kernel_copy_zone(
    RGBPixel* input, RGBPixel* zone_data, rect zone, size_t range, const uint fd_width, size_t start_zone_id)
{
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < range)
    {
        size_t x = id % zone.w;
        size_t y = id / zone.w;
        size_t fd_id = start_zone_id + (y * fd_width) + x;
        zone_data[id] = input[fd_id];
    }
}

/**
 * @brief Normalize an rgb image by dividing each pixel component by the given average.
 * @param image A contiguous rgb pixel buffer that represent the image.
 * @param image_res The number of pixel in the image.
 * @param rgb_average The average RGB components used to normalize the image.
 * @param stream The cuda stream used.
 */
void normalize_rgb_image(RGBPixel* image, uint image_res, RGBPixel rgb_average, cudaStream_t stream)
{
    RGBPixel* begin = image;
    RGBPixel* end = begin + image_res;
    RGBPixel* result = begin; // In-place transform.
    auto normalize = [rgb_average] __device__(RGBPixel val)
    {
        val.r /= rgb_average.r;
        val.g /= rgb_average.g;
        val.b /= rgb_average.b;
        return val * 1000;
    };

    auto execution_policy = thrust::cuda::par.on(stream);
    thrust::transform(execution_policy, begin, end, result, normalize);
}

/**
 * @brief Compute and apply the normalized weights of rgb colors
 * @param output The rgb float buffer
 * @param fd_height The height of the buffer
 * @param fd_width The width of the buffer
 * @param selection The selected zone
 * @param pixel_depth The depth of the pixel (usually 3)
 * @param averages The rgb averages to fill, used in UI
 * @param stream The used cuda stream
 */
void postcolor_normalize(float* output,
                              const uint fd_height,
                              const uint fd_width,
                              holovibes::units::RectFd selection,
                              const uchar pixel_depth,
                              float* averages,
                              const cudaStream_t stream)
{

    RGBPixel* rgb_output = (RGBPixel*)output;

    // Create zone.
    rect zone = {selection.x(), selection.y(), selection.unsigned_width(), selection.unsigned_height()};
    size_t frame_res = fd_height * fd_width;
    check_zone(zone, frame_res, fd_width);
    size_t zone_size = zone.h * zone.w;

    // ==============================================================
    // ======= Copy zone in contiguous memory buffer if needed ======
    // ==============================================================

    RGBPixel* gpu_zone_data;
    bool zone_selected = !(zone.x == 0 && zone.y == 0 && frame_res == zone_size);

    // No need to copy.
    if (!zone_selected)
    {
        gpu_zone_data = rgb_output;
    }
    // Copy in contiguous buffer.
    else
    {
        cudaXMalloc(&gpu_zone_data, zone_size * sizeof(RGBPixel));

        const uint threads = get_max_threads_1d();
        uint blocks = map_blocks_to_problem(zone_size, threads);

        size_t start_zone_id = fd_width * zone.y + zone.x;
        kernel_copy_zone<<<blocks, threads, 0, stream>>>(rgb_output,
                                                         gpu_zone_data,
                                                         zone,
                                                         zone_size,
                                                         fd_width,
                                                         start_zone_id);
        cudaCheckError();
    }

    // =====================================================================================
    // ====== Get the average RGB value of the zone (stored in the contiguous buffer) ======
    // =====================================================================================
    auto execution_policy = thrust::cuda::par.on(stream);
    auto add = [] __device__(RGBPixel acc, RGBPixel val) { return acc + val; };
    RGBPixel acc = {0, 0, 0};
    acc = thrust::reduce(execution_policy, gpu_zone_data, gpu_zone_data + zone_size, acc, add);
    acc = acc / zone_size;

    // Update averages for the GUI.
    averages[0] = acc.r;
    averages[1] = acc.g;
    averages[2] = acc.b;

    // Apply the normalization on the image
    normalize_rgb_image(rgb_output, frame_res, *((RGBPixel*)averages), stream);

    if (zone_selected)
    {
        cudaXFree(gpu_zone_data);
    }
}