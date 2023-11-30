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
        dst[id * 3 + HSV::H] = nR * 65536;
        dst[id * 3 + HSV::S] = nG * 65536;
        dst[id * 3 + HSV::V] = nB * 65536;
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

template <typename FUNC>
__global__ void kernel_compute_sum_depth(
    const cuComplex* input, float* output, size_t frame_res, size_t min_index, size_t max_index, FUNC func)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float res = 0.0f;

        for (size_t z = min_index; z <= max_index; ++z)
        {
            float input_elm = fabsf(input[z * frame_res + id].x);

            res += input_elm * func(z);
        }

        const size_t range = max_index - min_index + 1;
        output[id] = (res / (float)range);
    }
}

/// @brief Compute the sum depth of a pixel : sum of input[z] * func(z) between z1 and z2
/// @param input The input cuComplex buffer
/// @param output The output float buffer
/// @param frame_res The total number of pixels in one frame
/// @param min_index z1
/// @param max_index z2
/// @param func the function to call on z
template <typename FUNC>
void compute_sum_depth(const cuComplex* input,
                       float* output,
                       size_t frame_res,
                       size_t min_index,
                       size_t max_index,
                       FUNC func,
                       const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_compute_sum_depth<<<blocks, threads, 0, stream>>>(input, output, frame_res, min_index, max_index, func);
    cudaCheckError();
}

void compute_and_fill_h(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_h_index = hsv_struct.h.frame_index.min;
    const uint max_h_index = hsv_struct.h.frame_index.max;

    float* gpu_h_output = gpu_output + HSV::H * frame_res;

    // Hue is the moment 1 (average)
    auto func_moment_one = [] __device__(size_t z) -> size_t { return z; };

    compute_sum_depth(gpu_input, gpu_h_output, frame_res, min_h_index, max_h_index, func_moment_one, stream);
}

void compute_and_fill_s(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.max : hsv_struct.h.frame_index.max;

    float* gpu_s_output = gpu_output + HSV::S * frame_res;

    // Saturation is the moment 2 (variance)
    auto func_moment_two = [] __device__(size_t z) -> size_t { return z * z; };

    compute_sum_depth(gpu_input, gpu_s_output, frame_res, min_s_index, max_s_index, func_moment_two, stream);
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

    // Value is the moment 0
    auto func_moment_zero = [] __device__(size_t z) -> size_t { return 1; };

    compute_sum_depth(gpu_input, gpu_v_output, frame_res, min_v_index, max_v_index, func_moment_zero, stream);
}

/// @brief Compute the hsv values of each pixel, each channel use his own lambda function that describe the calculus
/// done on z
void compute_and_fill_hsv(const cuComplex* gpu_input,
                          float* gpu_output,
                          const size_t frame_res,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream)
{
    compute_and_fill_h(gpu_input, gpu_output, frame_res, hsv_struct, stream);
    compute_and_fill_s(gpu_input, gpu_output, frame_res, hsv_struct, stream);
    compute_and_fill_v(gpu_input, gpu_output, frame_res, hsv_struct, stream);
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

void apply_operations(float* gpu_arr,
                      uint height,
                      uint width,
                      float* const gpu_min,
                      float* const gpu_max,
                      const holovibes::CompositeChannel& channel_struct,
                      HSV channel,
                      const cudaStream_t stream)
{
    const uint frame_res = height * width;
    float* gpu_channel_arr = gpu_arr + frame_res * channel;

    apply_percentile_and_threshold(gpu_channel_arr,
                                   frame_res,
                                   width,
                                   height,
                                   channel_struct.threshold.min,
                                   channel_struct.threshold.max,
                                   stream);

    hsv_normalize(gpu_channel_arr, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom(gpu_channel_arr,
                         channel_struct.slider_threshold.min,
                         channel_struct.slider_threshold.max,
                         frame_res,
                         stream);

    hsv_normalize(gpu_channel_arr, frame_res, gpu_min, gpu_max, stream);
}

void apply_operations_on_h(float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const holovibes::CompositeHSV& hsv_struct,
                           const cudaStream_t stream)
{
    const uint frame_res = height * width;

    apply_operations(gpu_arr, height, width, gpu_min, gpu_max, hsv_struct.h, HSV::H, stream);

    // H channel has a blur option
    if (hsv_struct.h.blur.enabled)
    {
        apply_blur(gpu_arr, height, width, hsv_struct.h.blur.kernel_size, stream);
    }

    hsv_normalize(gpu_arr, frame_res, gpu_min, gpu_max, stream);
}

void apply_operations_on_s(float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const holovibes::CompositeHSV& hsv_struct,
                           const cudaStream_t stream)
{
    apply_operations(gpu_arr, height, width, gpu_min, gpu_max, hsv_struct.s, HSV::S, stream);
}

void apply_operations_on_v(float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const holovibes::CompositeHSV& hsv_struct,
                           const cudaStream_t stream)
{
    apply_operations(gpu_arr, height, width, gpu_min, gpu_max, hsv_struct.v, HSV::V, stream);
}

/// @brief Apply basic image processing operations on h,s and v (threshold, normalization, blur...)
void apply_operations_on_hsv(float* tmp_hsv_arr,
                             const uint height,
                             const uint width,
                             const holovibes::CompositeHSV& hsv_struct,
                             const cudaStream_t stream)
{
    // To perform a renormalization, a single min buffer and single max buffer is needed gpu side
    holovibes::cuda_tools::CudaUniquePtr<float> gpu_min(1);
    holovibes::cuda_tools::CudaUniquePtr<float> gpu_max(1);

    apply_operations_on_h(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
    apply_operations_on_s(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
    apply_operations_on_v(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
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
    compute_and_fill_hsv(gpu_input, tmp_hsv_arr, frame_res, hsv_struct, stream);

    apply_operations_on_hsv(tmp_hsv_arr, height, width, hsv_struct, stream);

    normalized_convert_hsv_to_rgb(tmp_hsv_arr, gpu_output, frame_res, stream);

    cudaXFree(tmp_hsv_arr);
}