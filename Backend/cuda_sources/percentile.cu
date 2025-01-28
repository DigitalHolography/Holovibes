#include "percentile.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include "cuda_memory.cuh"
#include "logger.hh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"

__global__ void copy_circle_kernel(const float* __restrict__ gpu_input,
                                   float* __restrict__ thrust_gpu_input_copy,
                                   const uint width,
                                   const uint height,
                                   const uint cx,
                                   const uint cy,
                                   const uint radius_squared,
                                   uint* counter)
{
    // Calculate the 2D indices for the current thread
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread operates within the image boundaries
    if (x >= width || y >= height)
        return;

    // Compute the squared distance from the center of the circle
    int dx = x - cx;
    int dy = y - cy;

    // Check if the pixel lies within the circular region
    if (dx * dx + dy * dy <= radius_squared)
    {
        uint index = atomicAdd(counter, 1);
        // Write the pixel value from the input image to the circular output buffer
        thrust_gpu_input_copy[index] = gpu_input[y * width + x];
    }
}

void fill_percentile_float_in_case_of_error(float* const out_percent, unsigned size_percent)
{
    for (size_t i = 0; i < size_percent; i++)
    {
        out_percent[i] = i;
    }
}

thrust::device_ptr<float> allocate_thrust(const uint frame_res, const cudaStream_t stream)
{
    float* raw_gpu_input_copy;
    // TODO: cudaXMallocAsync with the stream
    cudaSafeCall(cudaMalloc(&raw_gpu_input_copy, frame_res * sizeof(float)));
    return thrust::device_ptr<float>(raw_gpu_input_copy);
}

/*!
 * \brief Sort a copy of the array and save each of the values at h_percent % of the array in h_out_percent.
 *
 * Example:
 * `h_percent = [25f, 50f, 75f]`, `gpu_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`, `size_percent = 3`.
 * gives : `h_out_percent = [3, 6, 8]`
 *
 * \param thrust_gpu_input_copy The input array to sort
 * \param frame_res The res of a frame
 * \param h_percent The percentiles to compute
 * \param h_out_percent The output array
 * \param size_percent The size of the output array
 * \param stream The stream to use
 */
void compute_percentile(thrust::device_ptr<float>& thrust_gpu_input_copy,
                        const uint frame_res,
                        const float* const h_percent,
                        float* const h_out_percent,
                        const uint size_percent,
                        const cudaStream_t stream)
{
    thrust::sort(thrust::cuda::par.on(stream), thrust_gpu_input_copy, thrust_gpu_input_copy + frame_res);

    for (uint i = 0; i < size_percent; ++i)
    {
        const uint index = h_percent[i] / 100 * frame_res;
        cudaXMemcpyAsync(h_out_percent + i,
                         thrust_gpu_input_copy.get() + index,
                         sizeof(float),
                         cudaMemcpyDeviceToHost,
                         stream);
    }
    cudaXStreamSynchronize(stream);
}

uint calculate_frame_res(const uint width,
                         const uint height,
                         const uint offset,
                         const uint factor,
                         const float scale,
                         const bool compute_on_sub_zone)
{
    // Calculate the area based on the circular region defined by the scale
    const uint radius = static_cast<uint>(scale * std::min(width, height) / 2);
    const uint frame_res = (compute_on_sub_zone && radius > 0)
                               ? (3.14159f * radius * radius * 1.1f) // aproximation to avoid memory overflow
                               : width * height - 2 * offset * factor;
    CHECK(frame_res > 0);
    return frame_res;
}

uint calculate_frame_res(const uint width, const uint height, const uint offset, const uint factor)
{
    uint frame_res = width * height - 2 * offset * factor;
    CHECK(frame_res > 0);
    return frame_res;
}

/*!
 * \brief Calculate frame_res according to the width, height and required offset
 *
 * \param width The width of the frame
 * \param height The height of the frame
 * \param offset The offset
 * \param factor Multiplication factor for the offset (width for xz and height for yz)
 * \param scale The scale of the reticle
 * \param compute_on_sub_zone Whether to compute the percentile on the sub zone
 */
void compute_percentile_xy_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const float scale,
                                const bool compute_on_sub_zone,
                                const cudaStream_t stream)
{
    // Calculate the frame resolution based on the input dimensions and whether a sub-zone is used
    uint frame_res = calculate_frame_res(width, height, offset, width, scale, compute_on_sub_zone);
    offset *= width; // Adjust offset to account for row-based indexing

    thrust::device_ptr<float> thrust_gpu_input_copy(nullptr); // Pointer for GPU memory to store filtered data
    uint* d_counter = nullptr; // Device-side counter to track the number of pixels in the circular region
    uint h_counter = 0;        // Host-side counter to retrieve the result from the device

    try
    {
        // Allocate memory for the data buffer and the counter on the GPU
        thrust_gpu_input_copy = allocate_thrust(frame_res, stream);
        cudaSafeCall(cudaMalloc(&d_counter, sizeof(uint)));
        cudaSafeCall(cudaMemsetAsync(d_counter, 0, sizeof(uint), stream)); // Initialize the counter to 0

        if (compute_on_sub_zone)
        {
            // Parameters for the reticle zone
            const uint cx = width / 2;  // Center of the reticle (x-coordinate)
            const uint cy = height / 2; // Center of the reticle (y-coordinate)
            const uint radius = static_cast<uint>(scale * std::min(width, height) / 2); // Radius of the reticle
            const uint radius_squared = radius * radius;

            dim3 block_dim(16, 16);
            dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

            // Launch the kernel to copy pixels in the circular region
            copy_circle_kernel<<<grid_dim, block_dim, 0, stream>>>(gpu_input,
                                                                   thrust_gpu_input_copy.get(),
                                                                   width,
                                                                   height,
                                                                   cx,
                                                                   cy,
                                                                   radius_squared,
                                                                   d_counter);

            // Check for any errors during kernel execution
            cudaSafeCall(cudaGetLastError());

            // Synchronize the stream to ensure kernel execution is complete
            cudaSafeCall(cudaStreamSynchronize(stream));

            // Retrieve the actual number of copied pixels from the device counter
            cudaSafeCall(cudaMemcpyAsync(&h_counter, d_counter, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            cudaSafeCall(cudaStreamSynchronize(stream));
        }
        else
        {
            // Copy the entire image if no specific sub-zone is required
            thrust::copy(thrust::cuda::par.on(stream),
                         gpu_input + offset,
                         gpu_input + offset + frame_res,
                         thrust_gpu_input_copy);

            h_counter = frame_res; // Set counter to the full frame resolution
        }

        // Compute the percentiles on the filtered/circular data
        compute_percentile(thrust_gpu_input_copy, h_counter, h_percent, h_out_percent, size_percent, stream);
    }
    catch (const std::exception& e)
    {
        // Log critical errors and provide a fallback mechanism
        LOG_CRITICAL("{}", e.what());
        LOG_WARN("[Thrust] Error while computing a percentile");
        fill_percentile_float_in_case_of_error(h_out_percent, size_percent); // Fallback: fill percentiles with defaults
    }

    // Free GPU resources
    if (d_counter != nullptr)
        cudaSafeCall(cudaFree(d_counter));
    if (thrust_gpu_input_copy.get() != nullptr)
        cudaSafeCall(cudaFreeAsync(thrust_gpu_input_copy.get(), stream));
}

void compute_percentile_yz_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const float scale,
                                const bool compute_on_sub_zone,
                                const cudaStream_t stream)
{
    uint frame_res = calculate_frame_res(width, height, offset, height);

    thrust::device_ptr<float> thrust_gpu_input_copy(nullptr);
    try
    {
        thrust_gpu_input_copy = allocate_thrust(frame_res, stream);

        // Copy sub array (skip the 2 first columns and the 2 last columns)
        cudaSafeCall(cudaMemcpy2DAsync(thrust_gpu_input_copy.get(),          // dst
                                       (width - 2 * offset) * sizeof(float), // dpitch
                                       gpu_input + offset,                   // src
                                       width * sizeof(float),                // spitch
                                       (width - 2 * offset) * sizeof(float), // dwidth
                                       height,                               // dheight
                                       cudaMemcpyDeviceToDevice,             // kind
                                       stream));                             // stream

        compute_percentile(thrust_gpu_input_copy, frame_res, h_percent, h_out_percent, size_percent, stream);
    }
    catch (const std::exception& e)
    {
        LOG_CRITICAL("{}", e.what());
        LOG_WARN("[Thrust] Error while computing a percentile");
        fill_percentile_float_in_case_of_error(h_out_percent, size_percent);
    }
    if (thrust_gpu_input_copy.get() != nullptr)
        cudaSafeCall(cudaFreeAsync(thrust_gpu_input_copy.get(), stream));
}