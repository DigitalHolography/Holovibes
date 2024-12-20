#include <stdio.h>
#include <iostream>
#include <fstream>

#include "hsv.cuh"
#include "tools_hsv.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "percentile.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"
#include "reduce.cuh"
#include "unique_ptr.hh"
#include "logger.hh"

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

static constexpr ushort max_ushort_value = (1 << (sizeof(ushort) * 8)) - 1;

__global__ void kernel_normalized_convert_hsv_to_rgb(const float* src, float* dst, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float nNormalizedH = src[frame_res * HSV::H + id];
        float nNormalizedS = src[frame_res * HSV::S + id];
        float nNormalizedV = src[frame_res * HSV::V + id];
        float nR;
        float nG;
        float nB;
        if (nNormalizedS == 0.0F)
        {
            nR = nG = nB = nNormalizedV;
        }
        else
        {
            if (nNormalizedH == 1.0F)
                nNormalizedH = 0.0F;
            else
                nNormalizedH = nNormalizedH * 6.0F; // / 0.1667F
        }
        float nI = floorf(nNormalizedH);
        float nF = nNormalizedH - nI;
        float nM = nNormalizedV * (1.0F - nNormalizedS);
        float nN = nNormalizedV * (1.0F - nNormalizedS * nF);
        float nK = nNormalizedV * (1.0F - nNormalizedS * (1.0F - nF));
        if (nI == 0.0F)
        {
            nR = nNormalizedV;
            nG = nK;
            nB = nM;
        }
        else if (nI == 1.0F)
        {
            nR = nN;
            nG = nNormalizedV;
            nB = nM;
        }
        else if (nI == 2.0F)
        {
            nR = nM;
            nG = nNormalizedV;
            nB = nK;
        }
        else if (nI == 3.0F)
        {
            nR = nM;
            nG = nN;
            nB = nNormalizedV;
        }
        else if (nI == 4.0F)
        {
            nR = nK;
            nG = nM;
            nB = nNormalizedV;
        }
        else if (nI == 5.0F)
        {
            nR = nNormalizedV;
            nG = nM;
            nB = nN;
        }
        dst[id * 3 + HSV::H] = nR * max_ushort_value;
        dst[id * 3 + HSV::S] = nG * max_ushort_value;
        dst[id * 3 + HSV::V] = nB * max_ushort_value;
    }
}

/// @brief Convert an array of HSV normalized float [0,1] to an array of RGB float [0,65536]
/// @param src Input hsv array (contiguous pixel on x: [h1,...,hn,s1,...,sn,v1,...,vn])
/// @param dst Output rgb array (contiguous rgb channels: [r1,g1,b1,...,rn,gn,bn])
/// @param frame_res Total number of pixels on one frame
/// @param stream The used cuda stream
void normalized_convert_hsv_to_rgb(const float* src, float* dst, size_t frame_res, const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_normalized_convert_hsv_to_rgb<<<blocks, threads, 0, stream>>>(src, dst, frame_res);
    cudaCheckError();
}

/// @brief get the real z value, because of the FFT frequency shift
__device__ float get_real_z(size_t z, int depth, bool z_fft_shift)
{
    if (!z_fft_shift)
        return (float)z;

    if ((float)z < (float)depth / 2.0f)
        return (float)z;
    else if (z == depth / 2)
        return 0.0f;
    return (float)z - (float)depth;
}

/// @brief Convert the input complex number to a float
/// @param input_elm input complex number
/// @return a float, reprensenting the magnitude of the input complex number
__device__ float get_input_elm(cuComplex input_elm) { return hypotf(input_elm.x, input_elm.y); }

__global__ void kernel_compute_and_fill_h(const cuComplex* gpu_input,
                                          float* gpu_output,
                                          const size_t frame_res,
                                          const uint min_h_index,
                                          const uint max_h_index,
                                          int depth,
                                          bool z_fft_shift)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float num = 0.0f;
        float denom = 0.0f;

        // Compute the Average
        for (size_t z = min_h_index; z <= max_h_index; ++z)
        {
            const cuComplex* current_p_frame = gpu_input + (z * frame_res);
            float input_elm = get_input_elm(current_p_frame[id]);

            size_t real_z = get_real_z(z, depth, z_fft_shift);
            num += input_elm * real_z;
            denom += input_elm;
        }

        gpu_output[id] = (denom == 0.0f ? 0.0f : num / denom);
    }
}

__global__ void kernel_compute_and_fill_s(const cuComplex* gpu_input,
                                          float* gpu_output,
                                          const size_t frame_res,
                                          const uint min_s_index,
                                          const uint max_s_index,
                                          int depth,
                                          bool z_fft_shift)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        size_t z = min_s_index;
        float num = 0.0f;
        float denom = 0.0f;

        // Compute the Average
        for (z = min_s_index; z <= max_s_index; ++z)
        {
            const cuComplex* current_p_frame = gpu_input + (z * frame_res);
            float input_elm = get_input_elm(current_p_frame[id]);

            num += input_elm * get_real_z(z, depth, z_fft_shift);
            denom += input_elm;
        }
        float avg = (denom == 0.0f ? 0.0f : num / denom);

        // Compute the variance
        num = 0.0f;
        for (z = min_s_index; z <= max_s_index; ++z)
        {
            const cuComplex* current_p_frame = gpu_input + (z * frame_res);
            float input_elm = get_input_elm(current_p_frame[id]);

            float centered_z = get_real_z(z, depth, z_fft_shift) - avg;
            num += input_elm * centered_z * centered_z;
        }

        gpu_output[id] = (denom == 0.0f ? 0.0f : num / denom);
    }
}

__global__ void kernel_compute_and_fill_v(const cuComplex* gpu_input,
                                          float* gpu_output,
                                          const size_t frame_res,
                                          const uint min_v_index,
                                          const uint max_v_index)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float sum = 0.0f;

        // Compute the Sum
        for (size_t z = min_v_index; z <= max_v_index; ++z)
        {
            const cuComplex* current_p_frame = gpu_input + (z * frame_res);
            float input_elm = hypotf(current_p_frame[id].x, current_p_frame[id].y);
            sum += input_elm;
        }

        gpu_output[id] = sum;
    }
}

void compute_and_fill_h(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream,
                        int depth,
                        bool z_fft_shift)
{
    const uint min_h_index = hsv_struct.h.frame_index.min;
    const uint max_h_index = hsv_struct.h.frame_index.max;

    float* gpu_h_output = gpu_output + HSV::H * frame_res;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_compute_and_fill_h<<<blocks, threads, 0, stream>>>(gpu_input,
                                                              gpu_h_output,
                                                              frame_res,
                                                              min_h_index,
                                                              max_h_index,
                                                              depth,
                                                              z_fft_shift);
    cudaCheckError();
}

void compute_and_fill_s(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream,
                        int depth,
                        bool z_fft_shift)
{
    const uint min_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.max : hsv_struct.h.frame_index.max;

    float* gpu_s_output = gpu_output + HSV::S * frame_res;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_compute_and_fill_s<<<blocks, threads, 0, stream>>>(gpu_input,
                                                              gpu_s_output,
                                                              frame_res,
                                                              min_s_index,
                                                              max_s_index,
                                                              depth,
                                                              z_fft_shift);
    cudaCheckError();
}

void compute_and_fill_v(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.max : hsv_struct.h.frame_index.max;

    float* gpu_v_output = gpu_output + HSV::V * frame_res;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    kernel_compute_and_fill_v<<<blocks, threads, 0, stream>>>(gpu_input,
                                                              gpu_v_output,
                                                              frame_res,
                                                              min_v_index,
                                                              max_v_index);
    cudaCheckError();
}

/// @brief Compute the hsv values of each pixel of the frame
void compute_and_fill_hsv(const cuComplex* gpu_input,
                          float* gpu_output,
                          const size_t frame_res,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream,
                          int depth,
                          bool z_fft_shift)
{
    // HUE
    compute_and_fill_h(gpu_input, gpu_output, frame_res, hsv_struct, stream, depth, z_fft_shift);
    // SATURATION
    compute_and_fill_s(gpu_input, gpu_output, frame_res, hsv_struct, stream, depth, z_fft_shift);
    // VALUE
    compute_and_fill_v(gpu_input, gpu_output, frame_res, hsv_struct, stream);
}

/// @brief Basic operation on any specified channel (S or V)
void apply_operations(float* gpu_arr,
                      uint height,
                      uint width,
                      const holovibes::CompositeChannel& channel_struct,
                      HSV channel,
                      const cudaStream_t stream)
{
    const uint frame_res = height * width;
    float* gpu_channel_arr = gpu_arr + frame_res * channel;
    auto exec_policy = thrust::cuda::par.on(stream);

    // Apply the percentile and threshold on the specified channel
    apply_percentile_and_threshold(gpu_channel_arr,
                                   frame_res,
                                   width,
                                   height,
                                   channel_struct.threshold.min,
                                   channel_struct.threshold.max,
                                   stream);

    // Apply a clamping operation with the sliders min and max on the specified channel
    threshold_top_bottom(gpu_channel_arr,
                         channel_struct.slider_threshold.min,
                         channel_struct.slider_threshold.max,
                         frame_res,
                         stream);

    // The operation is a linear stretching of the values between the min and max sliders
    auto min = channel_struct.slider_threshold.min;
    auto scale = 1.0f / (channel_struct.slider_threshold.max - min);
    const auto op = [min, scale] __device__(const float pixel) { return (pixel - min) * scale; };

    // Apply the operation on the specified channel
    thrust::transform(exec_policy, gpu_channel_arr, gpu_channel_arr + frame_res, gpu_channel_arr, op);
}

/// @brief Special function for hue channel because hue has two UI sliders
void apply_operations_on_h(
    float* gpu_h_arr, uint height, uint width, const holovibes::CompositeH& h_struct, const cudaStream_t stream)
{
    const uint frame_res = height * width;
    auto exec_policy = thrust::cuda::par.on(stream);

    // Apply the percentile and threshold on the hue channel
    apply_percentile_and_threshold(gpu_h_arr,
                                   frame_res,
                                   width,
                                   height,
                                   h_struct.threshold.min,
                                   h_struct.threshold.max,
                                   stream);

    // Get the parameters from the two sliders on the GUI
    float range_min = h_struct.slider_threshold.min;
    float range_max = h_struct.slider_threshold.max;
    float shift_min = h_struct.slider_shift.min;
    float shift_max = h_struct.slider_shift.max;

    // m is the slope of the linear function, p is the offset
    auto m = (range_max - range_min) / (shift_max - shift_min);
    auto p = range_min - m * shift_min;

    // The operation is constant between 0 and shift_min, linear between shift_min and shift_max, and constant after
    const auto op = [m, p, shift_min, shift_max, range_min, range_max] __device__(const float pixel)
    {
        if (pixel < shift_min)
            return range_min;
        else if (pixel > shift_max)
            return range_max;
        else
            return m * pixel + p;
    };

    // Apply the operation on the hue channel
    thrust::transform(exec_policy, gpu_h_arr, gpu_h_arr + frame_res, gpu_h_arr, op);
}

/// @brief Apply basic image processing operations on h,s and v (threshold, normalization...)
void apply_operations_on_hsv(float* tmp_hsv_arr,
                             const uint height,
                             const uint width,
                             const holovibes::CompositeHSV& hsv_struct,
                             const cudaStream_t stream)
{
    // HUE
    apply_operations_on_h(tmp_hsv_arr, height, width, hsv_struct.h, stream);
    // SATURATION
    apply_operations(tmp_hsv_arr, height, width, hsv_struct.s, HSV::S, stream);
    // VALUE
    apply_operations(tmp_hsv_arr, height, width, hsv_struct.v, HSV::V, stream);
}

void hsv(const cuComplex* d_input,
         float* d_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct,
         bool checked)
{
    const uint frame_res = height * width;

    float* tmp_hsv_arr = nullptr;
    cudaSafeCall(cudaMalloc(&tmp_hsv_arr, frame_res * 3 * sizeof(float)));
    compute_and_fill_hsv(d_input, tmp_hsv_arr, frame_res, hsv_struct, stream, time_transformation_size, checked);

    apply_operations_on_hsv(tmp_hsv_arr, height, width, hsv_struct, stream);

    normalized_convert_hsv_to_rgb(tmp_hsv_arr, d_output, frame_res, stream);

    cudaXFree(tmp_hsv_arr);
}
