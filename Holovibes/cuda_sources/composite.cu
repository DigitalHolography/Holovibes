#include "cuda_memory.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "composite.cuh"
#include "map.cuh"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "logger.hh"

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
    auto normalize = [rgb_average] __host__ __device__(RGBPixel val)
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
    auto add = [] __host__ __device__(RGBPixel acc, RGBPixel val) { return acc + val; };
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