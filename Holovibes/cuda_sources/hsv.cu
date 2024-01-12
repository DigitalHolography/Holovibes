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

#define SAMPLING_FREQUENCY 1
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
__device__ float get_real_z(size_t z, int depth)
{
    if ((float)z < (float)depth / 2.0f)
        return (float)z;
    else if (z == depth / 2)
        return 0.0f;
    return (float)z - (float)depth;
}

/// @brief Convert the input complex number to a float
/// @param input_elm input complex number
/// @return a float, reprensenting the magnitude of the input complex number
__device__ float get_input_elm(cuComplex input_elm)
{
    return hypotf(input_elm.x, input_elm.y);
}

__global__ void kernel_compute_and_fill_h(const cuComplex* gpu_input,
                                          float* gpu_output,
                                          const size_t frame_res,
                                          const uint min_h_index,
                                          const uint max_h_index,
                                          int depth)
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

            size_t real_z = get_real_z(z, depth);
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
                                          int depth)
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

            num += input_elm * get_real_z(z, depth);
            denom += input_elm;
        }
        float avg = (denom == 0.0f ? 0.0f : num / denom);

        // Compute the variance
        num = 0.0f;
        for (z = min_s_index; z <= max_s_index; ++z)
        {
            const cuComplex* current_p_frame = gpu_input + (z * frame_res);
            float input_elm = get_input_elm(current_p_frame[id]);

            float centered_z = get_real_z(z, depth) - avg;
            num += input_elm * centered_z * centered_z;
        }

        gpu_output[id] = (denom == 0.0f ? 0.0f : num / denom);
    }
}

__global__ void kernel_compute_and_fill_v(const cuComplex* gpu_input,
                                          float* gpu_output,
                                          const size_t frame_res,
                                          const uint min_v_index,
                                          const uint max_v_index,
                                          int depth)
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
                        int depth)
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
                                                              depth);
    cudaCheckError();
}

void compute_and_fill_s(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream,
                        int depth)
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
                                                              depth);
    cudaCheckError();
}

void compute_and_fill_v(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream,
                        int depth)
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
                                                              max_v_index,
                                                              depth);
    cudaCheckError();
}

/// @brief Compute the hsv values of each pixel of the frame
void compute_and_fill_hsv(const cuComplex* gpu_input,
                          float* gpu_output,
                          const size_t frame_res,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream,
                          int depth)
{
    compute_and_fill_h(gpu_input, gpu_output, frame_res, hsv_struct, stream, depth);
    compute_and_fill_s(gpu_input, gpu_output, frame_res, hsv_struct, stream, depth);
    compute_and_fill_v(gpu_input, gpu_output, frame_res, hsv_struct, stream, depth);
}

// Apply a box blur on the specified array
void apply_blur(float* gpu_arr, uint height, uint width, float kernel_size, const cudaStream_t stream)
{
    size_t frame_res = height * width;

    float* gpu_float_blur_matrix;
    cudaSafeCall(cudaMalloc(&gpu_float_blur_matrix, frame_res * sizeof(float)));
    cudaSafeCall(cudaMemsetAsync(gpu_float_blur_matrix, 0, frame_res * sizeof(float), stream));

    float* blur_matrix;
    cudaSafeCall(cudaMallocHost(&blur_matrix, kernel_size * sizeof(float)));
    float blur_value = 1.0f / (float)(kernel_size * kernel_size);
    unsigned min_pos_kernel_y = height / 2 - kernel_size / 2;
    unsigned min_pos_kernel_x = width / 2 - kernel_size / 2;
    for (size_t i = 0; i < kernel_size; i++)
        blur_matrix[i] = blur_value;

    for (size_t i = 0; i < kernel_size; i++)
    {
        cudaXMemcpyAsync(gpu_float_blur_matrix + min_pos_kernel_x + width * (i + min_pos_kernel_y),
                         blur_matrix,
                         kernel_size * sizeof(float),
                         cudaMemcpyHostToDevice,
                         stream);
    }

    float* cpu_float_blur_matrix = new float[frame_res];
    cudaSafeCall(cudaMemcpyAsync(cpu_float_blur_matrix,
                                 gpu_float_blur_matrix,
                                 frame_res * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 stream));

    cuComplex* gpu_complex_blur_matrix;
    cudaSafeCall(cudaMalloc(&gpu_complex_blur_matrix, frame_res * sizeof(cuComplex)));
    cudaSafeCall(cudaMemcpy2DAsync(gpu_complex_blur_matrix,
                                   sizeof(cuComplex),
                                   gpu_float_blur_matrix,
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyDeviceToDevice,
                                   stream));

    shift_corners(gpu_complex_blur_matrix, 1, width, height, stream);

    CufftHandle handle{static_cast<int>(width), static_cast<int>(height), CUFFT_C2C};
    cufftSafeCall(cufftExecC2C(handle, gpu_complex_blur_matrix, gpu_complex_blur_matrix, CUFFT_FORWARD));

    cuComplex* gpu_cuComplex_buffer;
    cudaSafeCall(cudaMalloc(&gpu_cuComplex_buffer, frame_res * sizeof(cuComplex)));

    convolution_kernel(gpu_arr,
                       nullptr,
                       gpu_cuComplex_buffer,
                       &handle,
                       frame_res,
                       gpu_complex_blur_matrix,
                       false,
                       false,
                       stream);

    cudaXFree(gpu_cuComplex_buffer);
    cudaXFree(gpu_float_blur_matrix);
    cudaXFree(gpu_complex_blur_matrix);
}

void hsv_normalize(
    float* const gpu_arr, const uint frame_res, float* const gpu_min, float* const gpu_max, const cudaStream_t stream)
{
    reduce_min(gpu_arr, gpu_min, frame_res, stream); // Get the minimum value
    reduce_max(gpu_arr, gpu_max, frame_res, stream); // Get the maximum value

    const auto lambda = [gpu_min, gpu_max] __device__(const float pixel)
    { return (pixel - *gpu_min) * (1 / (*gpu_max - *gpu_min)); };

    auto exec_policy = thrust::cuda::par.on(stream);
    thrust::transform(exec_policy, gpu_arr, gpu_arr + frame_res, gpu_arr, lambda);
}

/// @brief Basic operation on any specified channel, with any operation requested
void apply_operations(float* gpu_arr,
                      uint height,
                      uint width,
                      const holovibes::CompositeChannel& channel_struct,
                      HSV channel,
                      threshold_op op,
                      const cudaStream_t stream)
{
    const uint frame_res = height * width;
    float* gpu_channel_arr = gpu_arr + frame_res * channel;
    auto exec_policy = thrust::cuda::par.on(stream);

    apply_percentile_and_threshold(gpu_channel_arr,
                                   frame_res,
                                   width,
                                   height,
                                   channel_struct.threshold.min,
                                   channel_struct.threshold.max,
                                   stream);

    if (op == CLAMP || op == CRUSH)
    {
        threshold_top_bottom(gpu_channel_arr,
                             channel_struct.slider_threshold.min,
                             channel_struct.slider_threshold.max,
                             frame_res,
                             stream);

        if (op == CRUSH)
        {
            auto min = channel_struct.slider_threshold.min;
            auto scale = 1.0f / (channel_struct.slider_threshold.max - min);
            const auto crush_op = [min, scale] __device__(const float pixel) { return (pixel - min) * scale; };

            thrust::transform(exec_policy, gpu_channel_arr, gpu_channel_arr + frame_res, gpu_channel_arr, crush_op);
        }
    }
    else if (op == ZOOM)
    {
        auto min = channel_struct.slider_threshold.min;
        auto diff = channel_struct.slider_threshold.max - min;
        const auto zoom_op = [min, diff] __device__(const float pixel) { return (pixel * diff) + min; };

        thrust::transform(exec_policy, gpu_channel_arr, gpu_channel_arr + frame_res, gpu_channel_arr, zoom_op);
    }
}

/// @brief Special function for hue channel because hue has two UI sliders and a blur option
void apply_operations_on_h(
    float* gpu_h_arr, uint height, uint width, const holovibes::CompositeH& h_struct, const cudaStream_t stream)
{
    // To perform a renormalization, a single min buffer and single max buffer is needed gpu side
    holovibes::cuda_tools::CudaUniquePtr<float> gpu_min(1);
    holovibes::cuda_tools::CudaUniquePtr<float> gpu_max(1);
    const uint frame_res = height * width;
    auto exec_policy = thrust::cuda::par.on(stream);

    // H channel has a blur option
    if (h_struct.blur.enabled)
    {
        apply_blur(gpu_h_arr, height, width, h_struct.blur.kernel_size, stream);

        //hsv_normalize(gpu_h_arr, height * width, gpu_min.get(), gpu_max.get(), stream);
    }

    apply_percentile_and_threshold(gpu_h_arr,
                                   frame_res,
                                   width,
                                   height,
                                   h_struct.threshold.min,
                                   h_struct.threshold.max,
                                   stream);

    float range_min = h_struct.slider_threshold.min;
    float range_max = h_struct.slider_threshold.max;
    float shift_min = h_struct.slider_shift.min;
    float shift_max = h_struct.slider_shift.max;

    auto m = (range_max - range_min) / (shift_max - shift_min);
    auto p = range_min - m * shift_min;
    const auto op = [m, p, shift_min, shift_max, range_min, range_max] __device__(const float pixel)
    {
        if (pixel < shift_min)
            return range_min;
        else if (pixel > shift_max)
            return range_max;
        else
            return m * pixel + p;
    };
    thrust::transform(exec_policy, gpu_h_arr, gpu_h_arr + frame_res, gpu_h_arr, op);
}

/// @brief Apply basic image processing operations on h,s and v (threshold, normalization, blur...)
void apply_operations_on_hsv(float* tmp_hsv_arr,
                             const uint height,
                             const uint width,
                             const holovibes::CompositeHSV& hsv_struct,
                             const cudaStream_t stream)
{
    // HUE
    apply_operations_on_h(tmp_hsv_arr, height, width, hsv_struct.h, stream);
    // SATURATION
    apply_operations(tmp_hsv_arr, height, width, hsv_struct.s, HSV::S, threshold_op::CRUSH, stream);
    // VALUE
    apply_operations(tmp_hsv_arr, height, width, hsv_struct.v, HSV::V, threshold_op::CRUSH, stream);
}

/// @brief Create rgb color by using hsv computation and then converting to rgb
/// @param gpu_input complex input buffer, on gpu side, size = width * height * time_transformation_size
/// @param gpu_output float output buffer, on gpu side, size = width * height * 3
/// @param width Width of the frame
/// @param height Height of the frame
/// @param stream Cuda stream used
/// @param time_transformation_size Depth of the frame cube
/// @param hsv_struct Struct containing all the UI parameters
void hsv(const cuComplex* gpu_input,
         float* gpu_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct)
{
    const uint frame_res = height * width;

    float* tmp_hsv_arr = nullptr;
    cudaSafeCall(cudaMalloc(&tmp_hsv_arr, frame_res * 3 * sizeof(float)));
    compute_and_fill_hsv(gpu_input, tmp_hsv_arr, frame_res, hsv_struct, stream, time_transformation_size);

    apply_operations_on_hsv(tmp_hsv_arr, height, width, hsv_struct, stream);

    normalized_convert_hsv_to_rgb(tmp_hsv_arr, gpu_output, frame_res, stream);

    cudaXFree(tmp_hsv_arr);
}

/*
__global__ void kernel_fill_hsv_xz_cut(const float* gpu_in_cut,
                                       float* gpu_hsv_cut,
                                       size_t width,
                                       uint depth,
                                       size_t frame_res,
                                       const holovibes::CompositeHSV& hsv_struct)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < width)
    {
        uint min_h_index = hsv_struct.h.frame_index.min;
        uint max_h_index = hsv_struct.h.frame_index.max;
        auto min = hsv_struct.h.slider_threshold.min;
        auto max = hsv_struct.h.slider_threshold.max;

        float num = 0.0f;
        float denom = 0.0f;
        for (size_t z = 0; z <= depth; ++z)
        {
            size_t index = width * z + id;
            float input_elm = gpu_in_cut[index];

            // For hsv cut colorization
            if (z >= min_h_index && z <= max_h_index)
            {
                float raw_hue = (float)(z - min_h_index) / (float)(max_h_index - min_h_index);

                if (raw_hue < min || raw_hue > max)
                {
                    // If the hue is outside the threshold, set the hue to red
                    gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
                }
                else
                {
                    // Else, normalize the hue
                    auto scale = 1.0f / (max - min);
                    gpu_hsv_cut[HSV::H * frame_res + index] = (raw_hue - min) * scale;
                }

                // Not too saturated to see the average
                gpu_hsv_cut[HSV::S * frame_res + index] = 0.5f;
            }
            else
            {
                // Greyscale
                gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
                gpu_hsv_cut[HSV::S * frame_res + index] = 0.0f;
            }
            // Normalize the value to [0,1]
            gpu_hsv_cut[HSV::V * frame_res + index] = input_elm / max_ushort_value;

            // For average computation
            num += input_elm * z;
            denom += input_elm;
        }

        // Set the average index to a saturated red
        float avg = (denom == 0.0f ? 0.0f : num / denom);
        size_t avg_index = std::round(avg);
        size_t index = width * avg_index + id;
        gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
        gpu_hsv_cut[HSV::S * frame_res + index] = 1.0f;
        gpu_hsv_cut[HSV::V * frame_res + index] = 1.0f;
    }
}

void fill_hsv_xz_cut(const float* gpu_in_cut,
                     float* gpu_hsv_cut,
                     uint width,
                     uint depth,
                     const holovibes::CompositeHSV& hsv_struct,
                     const cudaStream_t stream)
{
    size_t frame_res = width * depth;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width, threads);
    kernel_fill_hsv_xz_cut<<<blocks, threads, 0, stream>>>(gpu_in_cut,
                                                           gpu_hsv_cut,
                                                           width,
                                                           depth,
                                                           frame_res,
                                                           hsv_struct);
    cudaCheckError();
}

__global__ void kernel_fill_hsv_gradient(float* gpu_hsv_cut, size_t width, uint height)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < width * height)
    {
        size_t x = id % width;
        size_t y = id / width;
        float x_ratio = (float)x / (float)width;
        float y_ratio = (float)y / (float)height;
        gpu_hsv_cut[HSV::H * width * height + id] = y_ratio;
        gpu_hsv_cut[HSV::S * width * height + id] = 1.0f;
        gpu_hsv_cut[HSV::V * width * height + id] = x_ratio;
    }
}

void fill_hsv_gradient(float* gpu_hsv_cut, uint width, uint height, const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width, threads);
    kernel_fill_hsv_gradient<<<blocks, threads, 0, stream>>>(gpu_hsv_cut, width, height);
    cudaCheckError();
}

void fill_hsv_yz_cut(
    const float* gpu_in_cut, float* gpu_hsv_cut, uint height, uint depth, const holovibes::CompositeHSV& hsv_struct)
{
    size_t frame_res = height * depth;
    for (size_t i = 0; i < height; i++)
    {
        uint min_h_index = hsv_struct.h.frame_index.min;
        uint max_h_index = hsv_struct.h.frame_index.max;

        float num = 0.0f;
        float denom = 0.0f;
        for (size_t z = 0; z <= depth; ++z)
        {
            size_t index = depth * i + z;
            float input_elm = gpu_in_cut[index];

            // For hsv cut colorization
            if (z >= min_h_index && z <= max_h_index)
            {
                float raw_hue = (float)(z - min_h_index) / (float)(max_h_index - min_h_index);
                auto min = hsv_struct.h.slider_threshold.min;
                auto max = hsv_struct.h.slider_threshold.max;

                if (raw_hue < min || raw_hue > max)
                {
                    // If the hue is outside the threshold, set the hue to red
                    gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
                }
                else
                {
                    // Else, normalize the hue
                    auto scale = 1.0f / (max - min);
                    gpu_hsv_cut[HSV::H * frame_res + index] = (raw_hue - min) * scale;
                }

                // Not too saturated to see the average
                gpu_hsv_cut[HSV::S * frame_res + index] = 0.5f;
            }
            else
            {
                // Greyscale
                gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
                gpu_hsv_cut[HSV::S * frame_res + index] = 0.0f;
            }
            // Value doesn't change
            gpu_hsv_cut[HSV::V * frame_res + index] = input_elm / max_ushort_value;

            // For average computation
            num += input_elm * z;
            denom += input_elm;
        }

        // Set the average index to a saturated red
        float avg = (denom == 0.0f ? 0.0f : num / denom);
        size_t avg_index = std::round(avg);
        size_t index = depth * i + avg_index;
        gpu_hsv_cut[HSV::H * frame_res + index] = 0.0f;
        gpu_hsv_cut[HSV::S * frame_res + index] = 1.0f;
        gpu_hsv_cut[HSV::V * frame_res + index] = 1.0f;
    }
}

void hsv_cuts(const float* gpu_in_cut_xz,
              const float* gpu_in_cut_yz,
              float* gpu_out_cut_xz,
              float* gpu_out_cut_yz,
              uint width,
              uint height,
              int time_transformation_size,
              const holovibes::CompositeHSV& hsv_struct,
              const cudaStream_t stream)
{
    float* tmp_xz_hsv_arr = nullptr;
    // float* tmp_yz_hsv_arr = nullptr;
    cudaSafeCall(cudaMalloc(&tmp_xz_hsv_arr, width * time_transformation_size * 3 * sizeof(float)));
    // cudaSafeCall(cudaMalloc(&tmp_yz_hsv_arr, height * time_transformation_size * 3 * sizeof(float)));
    //fill_hsv_xz_cut(gpu_in_cut_xz, tmp_xz_hsv_arr, width, time_transformation_size, hsv_struct, stream);
    fill_hsv_gradient(tmp_xz_hsv_arr, width, time_transformation_size, stream);
    // fill_hsv_yz_cut(gpu_in_cut_yz, tmp_yz_hsv_arr, height, time_transformation_size, hsv_struct, stream);

    normalized_convert_hsv_to_rgb(tmp_xz_hsv_arr, gpu_out_cut_xz, width * time_transformation_size, stream);
    // normalized_convert_hsv_to_rgb(tmp_yz_hsv_arr, gpu_out_cut_yz, height * time_transformation_size, stream);

    cudaXFree(tmp_xz_hsv_arr);
    // cudaXFree(tmp_yz_hsv_arr);
}
*/