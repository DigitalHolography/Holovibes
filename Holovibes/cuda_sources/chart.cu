#include "chart.cuh"
#include "tools_conversion.cuh"
#include "units/rect.hh"
#include "unique_ptr.hh"
#include "tools.hh"
#include "cuda_memory.cuh"
#include "common.cuh"

#include <cstdio>
#include <cmath>

using holovibes::ChartPoint;
using holovibes::units::RectFd;

/*
 * Reduce a 32x32 tile
 * Each line is reduce at index 0 of the line
 *
 * Since 1 thread handles 2 pixels, we need to drop the 16 last threads
 * One could argue that everytime, the number of thread should be divided by 2
 * Our code does extra useless calculus but introducing branching inside a warp
 * calulation Would be slower than making some useless calculations like we are
 * doing
 *
 * No sync needed between += since we are using warps to reduce one line
 * But we need to tag the shared memory as volatile to avoid compiler reordering
 *
 */
static __device__ void reduce_full_width_tile(volatile float tile[32][32], const ushort x_tile, const ushort y_tile)
{
    // Reduce a line of 32 with a stride starting at 16
    if (x_tile < 16)
    {
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 16];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 8];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 4];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 2];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 1];
    }
    else // Drop 16 last threads
        return;
}

/*
 * Specific tile reduction to handle lines that are smaller than 32
 */
static __device__ void
reduce_width_tile(float tile[32][32], const ushort x_tile, const ushort y_tile, const ushort tile_width)
{
    // Stride should be tile width / 2, odd numbers forces the usage of ceil
    ushort stride = static_cast<ushort>(ceil(static_cast<float>(tile_width) / 2.0f));

    while (stride > 0)
    {
        if (x_tile + stride < tile_width)
            tile[y_tile][x_tile] += tile[y_tile][x_tile + stride];

        // We need to ceil for the calculations, but ceil(1/2) = 1 resulting in
        // an infinite loop
        if (stride == 1)
            return;

        // Again, to handle odd number, the stride should always be ceiled
        stride = static_cast<ushort>(ceil(static_cast<float>(stride) / 2.0f));
    }
}

/*
 * This kernel is used to make the sum of all pixel values of a zone in a frame
 *
 * 1 thread is started for each pixel of the zone
 * Thread (0,0) is located at the top left corner of the zone
 *
 * Each thread fill the local tile shared memory with the value of its pixel
 *
 * Each thread warp handles a line of the tile and executes a reduce to have the
 * sum at index 0 of the line
 *
 * Each first thread of the warp then writes its line cumulative sum inside the
 * first cell of the tile ([0][0])
 *
 * Then each tile writes it's cumulative sum in the total_sum output
 */
template <typename FUNC>
static __global__ void kernel_apply_mapped_zone_sum(const float* input,
                                                    const uint width,
                                                    double* total_sum,
                                                    const uint start_zone_x,
                                                    const uint start_zone_y,
                                                    const uint zone_width,
                                                    const uint zone_height,
                                                    FUNC element_map)
{
    // Position of the thread in the tile
    const uchar x_tile = threadIdx.x;
    const uchar y_tile = threadIdx.y;

    // Position of the thread in the zone
    const uint x_zone = blockIdx.x * blockDim.x + x_tile;
    const uint y_zone = blockIdx.y * blockDim.y + y_tile;

    /* Each 32x32 tile loads its image value in shared memory
       Shared memory is used to speed up the reduce speed */
    __shared__ float tile[32][32];

    // Boundary check: thread is inside the zone
    if (x_zone < zone_width && y_zone < zone_height)
    {
        // Each tile load its value from the image
        const uint x_image = x_zone + start_zone_x;
        const uint y_image = y_zone + start_zone_y;

        // No __syncthreads() is needed since we work at warp level, it's sync
        // by hardware constraint
        tile[y_tile][x_tile] = element_map(input[x_image + y_image * width]);

        // Last column might be smaller than 32
        const ushort last_column_tile_width = zone_width % blockDim.x;
        if (blockIdx.x != gridDim.x - 1 || last_column_tile_width == 0)
            reduce_full_width_tile(tile, x_tile, y_tile);
        else
            reduce_width_tile(tile, x_tile, y_tile, last_column_tile_width);

        // Wait for all lines to finish accumulating
        __syncthreads();
        /* x_tile == 0 because only the first thread of each line should write
           its accumulated value
           y_tile != 0 because the first line already has its accumalted value
           in tile[0][0] */
        if (x_tile == 0 && y_tile != 0)
        {
            // Accumulate every line result in the top left corner of the tile
            atomicAdd(&tile[0][0], tile[y_tile][0]);
        }
        // Wait for first thread of each line to accumulate in [0][0]
        __syncthreads();

        // Only first thread of each tile accumulate the tile result in the
        // output
        if (x_tile == 0 && y_tile == 0)
            atomicAdd(total_sum, static_cast<double>(tile[0][0]));
    }
}

template <typename FUNC>
void apply_mapped_zone_sum(const float* input,
                           const uint height,
                           const uint width,
                           double* output,
                           const RectFd& zone,
                           FUNC element_map,
                           const cudaStream_t stream)
{
    constexpr ushort block_width = 32;
    constexpr ushort block_height = 32;
    const dim3 block_size(block_width, block_height, 1);
    const dim3 grid_size(
        static_cast<ushort>(ceil(static_cast<float>(zone.width()) / static_cast<float>(block_width))),
        static_cast<ushort>(ceil(static_cast<float>(zone.height()) / static_cast<float>(block_height))),
        1);

    // Total sum of the zone
    kernel_apply_mapped_zone_sum<<<grid_size, block_size, 0, stream>>>(input,
                                                                       width,
                                                                       output,
                                                                       zone.topLeft().x(),
                                                                       zone.topLeft().y(),
                                                                       zone.width(),
                                                                       zone.height(),
                                                                       element_map);
    cudaCheckError();
}

void apply_zone_sum(const float* input,
                    const uint height,
                    const uint width,
                    double* output,
                    const RectFd& zone,
                    const cudaStream_t stream)
{
    static const auto identity_map = [] __device__(float val) { return val; };
    apply_mapped_zone_sum(input, height, width, output, zone, identity_map, stream);
}

static double
compute_average(float* input, const uint width, const uint height, const RectFd& zone, const cudaStream_t stream)
{
    holovibes::cuda_tools::CudaUniquePtr<double> gpu_sum_zone;
    if (!gpu_sum_zone.resize(1))
        return 1;

    cudaXMemsetAsync(gpu_sum_zone, 0.f, sizeof(double), stream);

    apply_zone_sum(input, height, width, gpu_sum_zone, zone, stream);

    double cpu_avg_zone;
    cudaXMemcpyAsync(&cpu_avg_zone, gpu_sum_zone, sizeof(double), cudaMemcpyDeviceToHost, stream);
    // Needs to synchronize since host memory is used after
    cudaXStreamSynchronize(stream);

    cpu_avg_zone /= (zone.height() * zone.width());

    return cpu_avg_zone;
}

void apply_zone_std_sum(const float* input,
                        const uint height,
                        const uint width,
                        double* output,
                        const RectFd& zone,
                        const double avg_zone,
                        const cudaStream_t stream)
{
    const auto std_map = [avg_zone] __device__(float val) { return (val - avg_zone) * (val - avg_zone); };
    apply_mapped_zone_sum(input, height, width, output, zone, std_map, stream);
}

static double compute_std(float* input,
                          const uint width,
                          const uint height,
                          const RectFd& zone,
                          const double cpu_avg_zone,
                          const cudaStream_t stream)
{
    holovibes::cuda_tools::CudaUniquePtr<double> gpu_std_sum_zone;
    if (!gpu_std_sum_zone.resize(1))
        return 1;

    cudaXMemsetAsync(gpu_std_sum_zone, 0.f, sizeof(double), stream);

    apply_zone_std_sum(input, height, width, gpu_std_sum_zone, zone, cpu_avg_zone, stream);

    double cpu_std_zone;
    cudaXMemcpyAsync(&cpu_std_zone, gpu_std_sum_zone, sizeof(double), cudaMemcpyDeviceToHost, stream);
    // Needs to synchronize since host memory is used after
    cudaXStreamSynchronize(stream);
    cpu_std_zone = sqrt(cpu_std_zone / (zone.height() * zone.width()));

    return cpu_std_zone;
}

ChartPoint make_chart_plot(float* input,
                           const uint width,
                           const uint height,
                           const RectFd& signal_zone,
                           const RectFd& noise_zone,
                           const cudaStream_t stream)
{
    double cpu_avg_signal = compute_average(input, width, height, signal_zone, stream);
    double cpu_avg_noise = compute_average(input, width, height, noise_zone, stream);

    double cpu_std_signal = compute_std(input, width, height, signal_zone, cpu_avg_signal, stream);

    return ChartPoint{
        cpu_avg_signal,
        cpu_avg_noise,
        cpu_avg_signal / cpu_avg_noise,
        10 * log10f(cpu_avg_signal / cpu_avg_noise),
        cpu_std_signal,
        cpu_std_signal / cpu_avg_noise,
        cpu_std_signal / cpu_avg_signal,
    };
}