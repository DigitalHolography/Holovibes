#include <stdio.h>
#include <iostream>
#include <fstream>

#include "hsv.cuh"
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

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#define SAMPLING_FREQUENCY 1

/*
 * \brief Convert an array of HSV normalized float to an array of RGB normalized
 * float i.e.: with "[  ]" a pixel: [HSV][HSV][HSV][HSV] -> [RGB][RGB][RGB][RGB]
 * NVdia function
 */
__global__ void kernel_normalized_convert_hsv_to_rgb(const float* src, float* dst, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        float nNormalizedH = src[id + frame_res * HSV::H];
        float nNormalizedS = src[id + frame_res * HSV::S];
        float nNormalizedV = src[id + frame_res * HSV::V];
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

void convert_hsv_to_rgb(const float* src, float* dst, size_t frame_res, const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_normalized_convert_hsv_to_rgb<<<blocks, threads, 0, stream>>>(src, dst, frame_res);
    cudaCheckError();
}

__global__ void
kernel_reduce_block_on_z(const float* gpu_input, float* gpu_output, const size_t frame_res, const size_t range)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= frame_res)
        return;

    float sum = 0.0f;
    for (size_t i = 0; i < range; i++)
    {
        sum += gpu_input[id * range + i];
    }

    gpu_output[id] = sum;
}

template <typename FUNC>
__global__ void kernel_multiply_by_frequencies(const float* gpu_input,
                                               float* products_output,
                                               const size_t zx_size,
                                               const size_t range,
                                               const size_t full_zx_size,
                                               const size_t full_range,
                                               const uint min_depth,
                                               const size_t buffer_size,
                                               FUNC unary_op)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= buffer_size)
        return;

    size_t y = id / zx_size;
    size_t zx_pos = id % zx_size;
    size_t x = zx_pos / range;
    size_t z = zx_pos % range;

    const size_t full_id = (y * full_zx_size) + (x * full_range) + min_depth + z;

    products_output[id] = unary_op(z, gpu_input[full_id]);
}

template <typename FUNC>
void multiply_and_reduce(const float* gpu_input,
                         float* gpu_output,
                         const size_t width,
                         const size_t frame_res,
                         const size_t full_range,
                         const uint min_index,
                         const uint max_index,
                         FUNC op,
                         const cudaStream_t stream)
{
    const uint range = max_index - min_index + 1;
    const size_t buffer_size = frame_res * range;

    float* gpu_products = nullptr;
    cudaSafeCall(cudaMalloc(&gpu_products, buffer_size * sizeof(float)));

    // Multiply the input data with the provided unary operation
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(buffer_size, threads);
    kernel_multiply_by_frequencies<<<blocks, threads, 0, stream>>>(gpu_input,
                                                                   gpu_products,
                                                                   width * range,
                                                                   range,
                                                                   width * full_range,
                                                                   full_range,
                                                                   min_index,
                                                                   buffer_size,
                                                                   op);
    cudaCheckError();

    // Reduce the input block on the depth
    blocks = map_blocks_to_problem(frame_res, threads);
    kernel_reduce_block_on_z<<<blocks, threads, 0, stream>>>(gpu_products, gpu_output, frame_res, range);
    cudaCheckError();

    cudaXFree(gpu_products);
}

void compute_and_fill_h(const float* gpu_input,
                        float* gpu_output,
                        const size_t width,
                        const size_t frame_res,
                        const size_t full_range,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_h_index = hsv_struct.h.frame_index.min;
    const uint max_h_index = hsv_struct.h.frame_index.max;

    float* gpu_h_output = gpu_output + (HSV::H * frame_res);

    auto op = [] __device__(size_t z, const float val) -> float { return val * z; };

    multiply_and_reduce(gpu_input, gpu_h_output, width, frame_res, full_range, min_h_index, max_h_index, op, stream);
}

void compute_and_fill_s(const float* gpu_input,
                        float* gpu_output,
                        const size_t width,
                        const size_t frame_res,
                        const size_t full_range,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.max : hsv_struct.h.frame_index.max;

    float* gpu_s_output = gpu_output + (HSV::S * frame_res);

    auto op = [] __device__(size_t z, const float val) -> float { return val * z * z; };

    multiply_and_reduce(gpu_input, gpu_s_output, width, frame_res, full_range, min_s_index, max_s_index, op, stream);
}

void compute_and_fill_v(const float* gpu_input,
                        float* gpu_output,
                        const size_t width,
                        const size_t frame_res,
                        const size_t full_range,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.max : hsv_struct.h.frame_index.max;

    float* gpu_v_output = gpu_output + (HSV::V * frame_res);

    auto op = [] __device__(size_t, const float val) -> float { return val; };

    multiply_and_reduce(gpu_input, gpu_v_output, width, frame_res, full_range, min_v_index, max_v_index, op, stream);
}

void compute_and_fill_hsv(const float* gpu_input,
                          float* gpu_output,
                          const size_t width,
                          const size_t frame_res,
                          const size_t full_range,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream)
{
    compute_and_fill_h(gpu_input, gpu_output, width, frame_res, full_range, hsv_struct, stream);
    compute_and_fill_s(gpu_input, gpu_output, width, frame_res, full_range, hsv_struct, stream);
    compute_and_fill_v(gpu_input, gpu_output, width, frame_res, full_range, hsv_struct, stream);
}

__global__ void threshold_top_bottom(float* output, const float tmin, const float tmax, const uint frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        output[id] = fminf(output[id], tmax);
        output[id] = fmaxf(output[id], tmin);
    }
}

__global__ void
kernel_from_distinct_components_to_interweaved_components(const float* src, float* dst, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        dst[id * 3] = src[id];
        dst[id * 3 + 1] = src[id + frame_res];
        dst[id * 3 + 2] = src[id + frame_res * 2];
    }
}

void from_distinct_components_to_interweaved_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_from_distinct_components_to_interweaved_components<<<blocks, threads, 0, stream>>>(src, dst, frame_res);
}

__global__ void
kernel_from_interweaved_components_to_distinct_components(const float* src, float* dst, size_t frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        dst[id] = src[id * 3];
        dst[id + frame_res] = src[id * 3 + 1];
        dst[id + frame_res * 2] = src[id * 3 + 2];
    }
}

void from_interweaved_components_to_distinct_components(const float* src,
                                                        float* dst,
                                                        size_t frame_res,
                                                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    kernel_from_interweaved_components_to_distinct_components<<<blocks, threads, 0, stream>>>(src, dst, frame_res);
}

void apply_percentile_and_threshold(float* gpu_arr,
                                    uint frame_res,
                                    uint width,
                                    uint height,
                                    float low_threshold,
                                    float high_threshold,
                                    const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_res, threads);
    float percent_out[2];
    const float percent_in_h[2] = {low_threshold, high_threshold};

    compute_percentile_xy_view(gpu_arr,
                               width,
                               height,
                               0,
                               percent_in_h,
                               percent_out,
                               2,
                               holovibes::units::RectFd(),
                               false,
                               stream);

    percent_out[0] = percent_in_h[0];
    percent_out[1] = percent_in_h[1];

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr, percent_out[0], percent_out[1], frame_res);
}

void apply_blur(
    float* gpu_arr, uint height, uint width, const holovibes::CompositeHSV& hsv_struct, const cudaStream_t stream)
{
    size_t frame_res = height * width;

    float* gpu_float_blur_matrix;
    cudaSafeCall(cudaMalloc(&gpu_float_blur_matrix, frame_res * sizeof(float)));
    cudaSafeCall(cudaMemsetAsync(gpu_float_blur_matrix, 0, frame_res * sizeof(float), stream));

    float* blur_matrix;
    const float kernel_size = hsv_struct.h.blur.kernel_size;
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
    cudaCheckError();

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
    map_generic(gpu_arr, gpu_arr, frame_res, lambda, stream);
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
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    /*
    apply_percentile_and_threshold(gpu_arr,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.h.threshold.min,
                                   hsv_struct.h.threshold.max,
                                   stream);
    */

    //map_multiply(gpu_arr, gpu_arr, frame_res, -1.0f, stream);
    hsv_normalize(gpu_arr, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr,
                                                         hsv_struct.h.slider_threshold.min,
                                                         hsv_struct.h.slider_threshold.max,
                                                         frame_res);

    if (hsv_struct.h.blur.enabled)
    {
        apply_blur(gpu_arr, height, width, hsv_struct, stream);
    }

    hsv_normalize(gpu_arr, frame_res, gpu_min, gpu_max, stream);
    //map_multiply(gpu_arr, gpu_arr, frame_res, 0.66f, stream);
}

void apply_operations_on_s(float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const holovibes::CompositeHSV& hsv_struct,
                           const cudaStream_t stream)
{
    const uint frame_res = height * width;
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    float* gpu_arr_s = gpu_arr + frame_res * HSV::S;

    /*
    apply_percentile_and_threshold(gpu_arr_s,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.s.threshold.min,
                                   hsv_struct.s.threshold.max,
                                   stream);
    */

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_s,
                                                         hsv_struct.s.slider_threshold.min,
                                                         hsv_struct.s.slider_threshold.max,
                                                         frame_res);

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);
}

void apply_operations_on_v(float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const holovibes::CompositeHSV& hsv_struct,
                           const cudaStream_t stream)
{
    const uint frame_res = height * width;
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    float* gpu_arr_v = gpu_arr + frame_res * HSV::V;

    /*
    apply_percentile_and_threshold(gpu_arr_v,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.v.threshold.min,
                                   hsv_struct.v.threshold.max,
                                   stream);
    */

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_v,
                                                         hsv_struct.v.slider_threshold.min,
                                                         hsv_struct.v.slider_threshold.max,
                                                         frame_res);

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);
}

__global__ void kernel_rotate_arr_to_contiguous_z(
    const cuComplex* gpu_input, float* rotated_arr, const uint frame_res, const uint width, const uint range)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t depth = id / frame_res;
    size_t frame_pos = id % frame_res;
    size_t y = frame_pos / width;
    size_t x = frame_pos % width;
    const size_t rotated_id = (y * range * width) + (x * range) + depth;

    float val = fabsf(gpu_input[id].x);
    rotated_arr[rotated_id] = val;
}

void rotate_arr_to_contiguous_z(const cuComplex* gpu_input,
                                float* rotated_arr,
                                const uint frame_res,
                                const uint width,
                                const uint range,
                                const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res * range, threads);

    kernel_rotate_arr_to_contiguous_z<<<blocks, threads, 0, stream>>>(gpu_input, rotated_arr, frame_res, width, range);
    cudaCheckError();
}

__global__ void set_to_gradient(float* output, size_t height, size_t width) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= height * width)
        return;

    size_t line = id / width;

    float value = ((float)line / (float)height);

    /*
    output[id * 3] = value;
    output[id * 3 + 1] = value;
    output[id * 3 + 2] = value;
    */
    output[id] = 0.0f;
    output[id + (width * height)] = 0.0f;
    output[id + 2 * (width * height)] = value;
}

void hsv(const cuComplex* gpu_input,
         float* gpu_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct)
{
    const uint frame_res = height * width;

    // HSV rotated array for contiguous data on z axis
    float* rotated_arr = nullptr;
    cudaSafeCall(cudaMalloc(&rotated_arr, frame_res * time_transformation_size * sizeof(float)));
    rotate_arr_to_contiguous_z(gpu_input, rotated_arr, frame_res, width, time_transformation_size, stream);

    float* tmp_hsv_arr = nullptr;
    cudaSafeCall(cudaMalloc(&tmp_hsv_arr, frame_res * sizeof(float) * 3));
    compute_and_fill_hsv(rotated_arr, tmp_hsv_arr, width, frame_res, time_transformation_size, hsv_struct, stream);

    {
        // To perform a renormalization, a single min buffer and single max buffer is needed gpu side
        holovibes::cuda_tools::UniquePtr<float> gpu_min(1);
        holovibes::cuda_tools::UniquePtr<float> gpu_max(1);

        apply_operations_on_h(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_s(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_v(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
    }

    /*
    float* hues = (float*) malloc(3 * sizeof(float));
    cudaXMemcpy(hues, tmp_hsv_arr, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("h1=%f, h2=%f, h3=%f\n", hues[0], hues[1], hues[2]);
    float* saturations = (float*) malloc(3 * sizeof(float));
    cudaXMemcpy(saturations, tmp_hsv_arr + frame_res, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("s1=%f, s2=%f, s3=%f\n", saturations[0], saturations[1], saturations[2]);
    */

    convert_hsv_to_rgb(tmp_hsv_arr, gpu_output, frame_res, stream);

    cudaXFree(rotated_arr);
    cudaXFree(tmp_hsv_arr);
}