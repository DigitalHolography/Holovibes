#include "chart.cuh"

#include "unique_ptr.hh"

using holovibes::ChartPoint;
using holovibes::units::RectFd;

#define TILE_SIZE 32
#define STRIDE_SIZE 16

/*! \brief Perform a reduction operation on a row of a 2D tile using a fixed stride size
 *
 * This function reduces the values in a single row of a 2D tile of size TILE_SIZE. The reduction is performed in a series of
 * steps where elements at a specific stride from each other are summed. The reduction happens in place, modifying the input tile.
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
 * \param[in,out] tile The 2D tile of size TILE_SIZE x TILE_SIZE to reduce
 * \param[in] x_tile The column index within the tile
 * \param[in] y_tile The row index within the tile
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

/*! \brief Specific tile reduction to handle lines that are smaller than 32
 *
 * This function reduces the values in a single row of a 2D tile of size TILE_SIZE. It handles cases where
 * the tile width is smaller than 32 by using a dynamically computed stride. The reduction happens in place, modifying the input tile.
 *
 * \param[in,out] tile The 2D tile of size TILE_SIZE x TILE_SIZE to reduce
 * \param[in] x_tile The column index within the tile
 * \param[in] y_tile The row index within the tile
 * \param[in] tile_width The width of the current tile to reduce, which is smaller than 32
 */
static __device__ void
reduce_width_tile(float tile[TILE_SIZE][TILE_SIZE], const ushort x_tile, const ushort y_tile, const ushort tile_width)
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
            atomicAdd(output, static_cast<double>(tile[0][0]));
    }
}

/*! \brief Apply a user-defined mapping function to a specific zone of input data and compute the sum
 *
 * This function launches a CUDA kernel that applies a mapping function to each element of a specific zone in the input data.
 * The mapped values are then summed and stored in the `output` variable. The kernel configuration uses a block size of 32x32 and
 * adapts the grid size based on the dimensions of the specified zone.
 *
 * \tparam FUNC A callable type (e.g., a lambda or function) representing the mapping function applied to each element
 * \param[out] output The device memory pointer where the total sum of the mapped values will be stored
 * \param[in] input The input buffer containing image data
 * \param[in] height The height of the input image
 * \param[in] width The width of the input image
 * \param[in] zone The region of interest (zone) where the mapping and summation will occur
 * \param[in] element_map A callable function that maps each element in the zone to a desired value
 * \param[in] stream The CUDA stream on which the kernel will be launched
 */
template <typename FUNC>
void apply_mapped_zone_sum(double* __restrict__ output,
                           const float* __restrict__ input,
                           const uint height,
                           const uint width,
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

/*! \brief Compute the sum of values in a specific zone of the input data
 *
 * This function computes the sum of all values within a specified rectangular zone of the input data.
 * It uses the `apply_mapped_zone_sum` function with an identity mapping, meaning the values themselves are summed without modification.
 *
 * \param[out] output The device memory pointer where the total sum of the zone will be stored
 * \param[in] input The input buffer containing image data
 * \param[in] height The height of the input image
 * \param[in] width The width of the input image
 * \param[in] zone The region of interest (zone) where the summation will occur
 * \param[in] stream The CUDA stream on which the operation will be performed
 */
void apply_zone_sum(double* __restrict__ output,
                    const float* __restrict__ input,
                    const uint height,
                    const uint width,
                    const RectFd& zone,
                    const cudaStream_t stream)
{
    static const auto identity_map = [] __device__(float val) { return val; };
    apply_mapped_zone_sum(output, input, height, width, zone, identity_map, stream);
}

static double
compute_average(float* __restrict__ input, const uint width, const uint height, const RectFd& zone, const cudaStream_t stream)
{
    holovibes::cuda_tools::CudaUniquePtr<double> gpu_sum_zone;
    if (!gpu_sum_zone.resize(1))
        return 1;

    cudaXMemsetAsync(gpu_sum_zone, 0.f, sizeof(double), stream);

    apply_zone_sum(gpu_sum_zone, input, height, width, zone, stream);

    double cpu_avg_zone;
    cudaXMemcpyAsync(&cpu_avg_zone, gpu_sum_zone, sizeof(double), cudaMemcpyDeviceToHost, stream);
    // Needs to synchronize since host memory is used after
    cudaXStreamSynchronize(stream);

    cpu_avg_zone /= (zone.height() * zone.width());

    return cpu_avg_zone;
}

/*! \brief Compute the average value of a specific zone within the input data
 *
 * This function calculates the average value of all pixels within a specified rectangular zone of the input data.
 * It first computes the sum of all values in the zone and then divides it by the number of pixels in the zone to obtain the average.
 *
 * \param[in] input The input buffer containing image data
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] zone The region of interest (zone) where the average will be computed
 * \param[in] stream The CUDA stream on which the computations will be launched
 * \return The computed average value for the specified zone
 */
void apply_zone_std_sum(double* __restrict__ output,
                        const float* __restrict__ input,
                        const uint height,
                        const uint width,
                        const RectFd& zone,
                        const double avg_zone,
                        const cudaStream_t stream)
{
    const auto std_map = [avg_zone] __device__(float val) { return (val - avg_zone) * (val - avg_zone); };
    apply_mapped_zone_sum(output, input, height, width, zone, std_map, stream);
}

/*! \brief Compute the standard deviation of the values in a specified zone of the input data
 *
 * \param[in] input The input buffer containing image data
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] zone The region of interest in the image for which the standard deviation is computed
 * \param[in] cpu_avg_zone The precomputed average value of the data within the specified zone
 * \param[in] stream The CUDA stream on which the computations will be launched
 * \return The computed standard deviation of the values within the specified zone
 */
static double compute_std(float* __restrict__ input,
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

    apply_zone_std_sum(gpu_std_sum_zone, input, height, width, zone, cpu_avg_zone, stream);

    double cpu_std_zone;
    cudaXMemcpyAsync(&cpu_std_zone, gpu_std_sum_zone, sizeof(double), cudaMemcpyDeviceToHost, stream);
    // Needs to synchronize since host memory is used after
    cudaXStreamSynchronize(stream);
    cpu_std_zone = sqrt(cpu_std_zone / (zone.height() * zone.width()));

    return cpu_std_zone;
}

ChartPoint make_chart_plot(float* __restrict__ input,
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
