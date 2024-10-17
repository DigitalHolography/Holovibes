#include "chart.cuh"

#include "unique_ptr.hh"

using holovibes::ChartPoint;
using holovibes::units::RectFd;

#define TILE_SIZE 32
#define STRIDE_SIZE 16

/*
 * Reduce a TILE_SIZExTILE_SIZE tile
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
static __device__ void
reduce_full_width_tile(volatile float tile[TILE_SIZE][TILE_SIZE], const ushort x_tile, const ushort y_tile)
{
    // Reduce a line of TILE_SIZE with a stride starting at STRIDE_SIZE
    if (x_tile < STRIDE_SIZE)
    {
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 16];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 8];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 4];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 2];
        tile[y_tile][x_tile] += tile[y_tile][x_tile + 1];
    }
}

/*!
 * \brief Performs a warp-level reduction to sum values across threads in a warp using warp-level primitives.
 *
 * This inline device function leverages CUDA's warp-level primitives to perform an efficient sum
 * reduction across all threads within a warp (typically 32 threads). Warp-level primitives, like
 * `__shfl_down_sync`, enable threads within a warp to communicate directly without the overhead of
 * shared memory or explicit synchronization, significantly optimizing reduction operations.
 *
 * The reduction is achieved by progressively halving the number of active threads in each iteration.
 * Starting with an offset equal to `warpSize / 2`, each thread exchanges its value with a neighbor
 * `offset` positions below, and accumulates the result. This continues until all values in the warp
 * are summed together. The function returns the final sum, and while every thread in the warp
 * computes this sum, typically only thread 0 will use the result.
 *
 * As highlighted in NVIDIA's CUDA blog on warp-level primitives
 * (https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/), this approach avoids the
 * need for full block-wide synchronization and the use of shared memory, making it faster and more
 * efficient for warp-level reductions.
 *
 * In the for loop:
 * - The initial `offset` is `warpSize / 2`, which corresponds to 16 threads (assuming `warpSize` is 32).
 * - The value of `offset` is halved after each iteration (16, 8, 4, 2, 1), allowing the threads to
 *   progressively accumulate values within the warp.
 *
 * The intrinsic `__shfl_down_sync` enables this reduction by shifting values between threads in a warp.
 * The mask `0xFFFFFFFF` ensures that all threads in the warp participate in the operation.
 *
 * \param[in] val The value held by the current thread, which will be summed across the warp.
 * \return The sum of the values across the warp. Although all threads in the warp return the same sum,
 *         usually only the first thread in each warp (thread 0) will use the final result.
 */
__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/*! \brief Kernel to compute the sum of pixel values within a zone in a frame
 *
 * This kernel assigns one thread to each pixel in a specified zone.
 * Thread (0,0) corresponds to the top-left corner of the zone.
 *
 * Each thread loads its corresponding pixel value into shared memory.
 *
 * A warp (32 threads) processes a row of the tile, performing a reduction to
 * store the sum in the first index of the row.
 *
 * The first thread in each warp writes the cumulative sum of its row to the
 * top-left tile element (tile[0][0]).
 *
 * Finally, the top-left element of the tile is written to the global output,
 * representing the sum of all pixels in the tile.
 *
 * \param[out] output Global memory location where the cumulative sum of the zone will be written
 * \param[in] input Input frame from which pixel values are read
 * \param[in] width Width of the input frame
 * \param[in] start_zone_x X-coordinate of the top-left corner of the zone
 * \param[in] start_zone_y Y-coordinate of the top-left corner of the zone
 * \param[in] zone_width Width of the zone to process
 * \param[in] zone_height Height of the zone to process
 * \param[in] element_map Mapping function to apply to each pixel before summing
 */
template <typename FUNC>
static __global__ void kernel_apply_mapped_zone_sum(double* __restrict__ output,
                                                    const float* __restrict__ input,
                                                    const uint width,
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

    /*
     * Each TILE_SIZExTILE_SIZE tile loads its image value in shared memory
     *   Shared memory is used to speed up the reduce speed
     */
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // Boundary check: thread is inside the zone
    if (x_zone < zone_width && y_zone < zone_height)
    {
        // Each tile load its value from the image
        const uint x_image = x_zone + start_zone_x;
        const uint y_image = y_zone + start_zone_y;

        /*
         * No explicit __syncthreads() is needed here because each warp operates on its own line,
         * and warp-level synchronization is automatically handled by the hardware.
         * The threads within a warp are inherently synchronized by the GPU architecture.
         */
        tile[y_tile][x_tile] = element_map(input[x_image + y_image * width]);

        // Each thread checks if he is in a valid area
        if (x_zone < zone_width)
            reduce_full_width_tile(tile, x_tile, y_tile);

        // Wait for all lines to finish accumulating
        __syncthreads();

        // Optimized reduction with warp-level primitives
        if (x_tile == 0) {
            float row_sum = warp_reduce_sum(tile[y_tile][0]);
            if (threadIdx.x == 0) {
                atomicAdd(&tile[0][0], row_sum);
            }
        }

        // Wait for first thread of each line to accumulate in [0][0]
        __syncthreads();

        // Only first thread of each tile accumulate the tile result in the
        // output
        if (x_tile == 0 && y_tile == 0)
            atomicAdd(output, static_cast<double>(tile[0][0]));
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
    kernel_apply_mapped_zone_sum<<<grid_size, block_size, 0, stream>>>(output,
                                                                       input,
                                                                       width,
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
