#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuda_memory.cuh"
#include <spdlog/spdlog.h>

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

/*
** \brief Sort a copy of the array and save each of the values at h_percent % of
*the array in h_out_percent
** i.e. h_percent = [25f, 50f, 75f] and gpu_input = [1, 2, 3, 4, 5, 6, 7, 8, 9,
*10] and size_percent = 3
** gives : h_out_percent = [3, 6, 8]
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

/*!
** \brief Calculate frame_res according to the width, height and required offset
*
* \param factor Multiplication factor for the offset (width for xz and height
*for yz)
*/
uint calculate_frame_res(const uint width,
                         const uint height,
                         const uint offset,
                         const uint factor,
                         const holovibes::units::RectFd& sub_zone,
                         const bool compute_on_sub_zone)
{
    // Sub_zone area might be equal to 0 if the overlay hasn't been loaded yet.
    // This is a dirty fix, but it mostly works
    uint frame_res =
        (compute_on_sub_zone && sub_zone.area() != 0) ? sub_zone.area() : width * height - 2 * offset * factor;
    CHECK(frame_res > 0);
    return frame_res;
}

uint calculate_frame_res(const uint width, const uint height, const uint offset, const uint factor)
{
    uint frame_res = width * height - 2 * offset * factor;
    CHECK(frame_res > 0);
    return frame_res;
}

void compute_percentile_xy_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const holovibes::units::RectFd& sub_zone,
                                const bool compute_on_sub_zone,
                                const cudaStream_t stream)
{
    uint frame_res = calculate_frame_res(width, height, offset, width, sub_zone, compute_on_sub_zone);
    offset *= width;

    thrust::device_ptr<float> thrust_gpu_input_copy(nullptr);
    try
    {
        thrust_gpu_input_copy = allocate_thrust(frame_res, stream);
        if (compute_on_sub_zone)
            frame_memcpy(thrust_gpu_input_copy.get(), gpu_input + offset, sub_zone, width, stream);
        else
            thrust::copy(thrust::cuda::par.on(stream),
                         gpu_input + offset,
                         gpu_input + offset + frame_res,
                         thrust_gpu_input_copy);

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

void compute_percentile_yz_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const holovibes::units::RectFd& sub_zone,
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
