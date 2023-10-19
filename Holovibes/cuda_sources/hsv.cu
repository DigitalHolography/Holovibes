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
        float nNormalizedH = src[id * 3];
        float nNormalizedS = src[id * 3 + 1];
        float nNormalizedV = src[id * 3 + 2];
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
        dst[id * 3] = nR;
        dst[id * 3 + 1] = nG;
        dst[id * 3 + 2] = nB;
    }
}

/**
 * @brief Fill the frequencies array with sequences of indexes, used for later operations
 * @param gpu_freq_arr The output frequencies array
 * @param time_transformation_size The depth of the input frame cube
 * @param stream The used cuda stream
 */
void fill_frequencies_arrays(float* gpu_freq_arr, const int time_transformation_size, const cudaStream_t stream)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    // Allocate two times the time_transformation_size, the first half will contain the sequence of indexes, the second
    // half the sequence of squared indexes
    cudaSafeCall(cudaMalloc(&gpu_freq_arr, time_transformation_size * sizeof(float) * 2));

    float* begin = gpu_freq_arr;
    float* mid = gpu_freq_arr + time_transformation_size;
    float* end = gpu_freq_arr + (time_transformation_size * 2);

    // Fill the first half of the array with a sequence from 0 to time_transformation_size - 1
    thrust::sequence(exec_policy, begin, mid);
    // Fill the second half of the array with a sequence from 0 to time_transformation_size - 1
    thrust::sequence(exec_policy, mid, end);

    // Square the second half of the array
    auto square_op = [] __host__ __device__(float val) { return val * val; };
    thrust::transform(exec_policy, mid, end, square_op);
}

/*
__device__ int fast_power(int base, int exponent)
{
    if (exponent == 0)
        return 1;
    if (exponent == 1)
        return base;

    int result = fast_power(base, exponent / 2);
    result *= result;

    if (exponent % 2 == 1)
        result *= base;

    return result;
}
*/

/**
 * @brief Compute the moment n of a pixel : sum of I(z) * z^n between z1 and z2
 * @param input The input cuComplex buffer
 * @param output The output float buffer
 * @param frame_res The total number of pixels in one frame
 * @param min_index z1
 * @param max_index z2
 * @param moment_n The exponent (n)
 * @param channel_index The index of the specified channel (H = 0, S = 1, V = 2)
 */
/*
__global__ void kernel_compute_moment(const cuComplex* input,
                                      float* output,
                                      const size_t frame_res,
                                      const size_t min_index,
                                      const size_t max_index,
                                      const size_t moment_n,
                                      const size_t channel_index)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        const size_t index_V = id * 3 + channel_index;
        output[index_V] = 0.0f;

        for (size_t z = min_index; z <= max_index; ++z)
        {
            float input_elm = fabsf(input[z * frame_res + id].x);

            output[index_V] += input_elm * fast_power(z, moment_n);
        }
    }
}
*/

/*
void compute_and_fill_h(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    const uint min_h_index = hsv_struct.h.frame_index.min;
    const uint max_h_index = hsv_struct.h.frame_index.max;

    // Hue is the moment 1 (average)
    kernel_compute_moment<<<blocks, threads, 0, stream>>>(gpu_input,
                                                          gpu_output,
                                                          frame_res,
                                                          min_h_index,
                                                          max_h_index,
                                                          1,
                                                          HSV::H);
}

void compute_and_fill_s(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    const uint min_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_s_index =
        hsv_struct.s.frame_index.activated ? hsv_struct.s.frame_index.max : hsv_struct.h.frame_index.max;

    // Saturation is the moment 2 (variance)
    kernel_compute_moment<<<blocks, threads, 0, stream>>>(gpu_input,
                                                          gpu_output,
                                                          frame_res,
                                                          min_s_index,
                                                          max_s_index,
                                                          2,
                                                          HSV::S);
}

void compute_and_fill_v(const cuComplex* gpu_input,
                        float* gpu_output,
                        const size_t frame_res,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    const uint min_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.min : hsv_struct.h.frame_index.min;
    const uint max_v_index =
        hsv_struct.v.frame_index.activated ? hsv_struct.v.frame_index.max : hsv_struct.h.frame_index.max;

    // Value is the moment 0
    kernel_compute_moment<<<blocks, threads, 0, stream>>>(gpu_input,
                                                          gpu_output,
                                                          frame_res,
                                                          min_v_index,
                                                          max_v_index,
                                                          0,
                                                          HSV::V);
}
*/

/*
void compute_and_fill_hsv(const cuComplex* gpu_input,
                          float* gpu_output,
                          const size_t frame_res,
                          float* gpu_omega_arr,
                          size_t omega_arr_size,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream)
{
    compute_and_fill_h(gpu_input, gpu_output, frame_res, hsv_struct, stream);
    compute_and_fill_s(gpu_input, gpu_output, frame_res, hsv_struct, stream);
    compute_and_fill_v(gpu_input, gpu_output, frame_res, hsv_struct, stream);

    cudaCheckError();
}
*/

/*
void multiply_and_reduce(const float* gpu_input,
                         float* multiplier_input,
                         float* products,
                         float* output,
                         const size_t range,
                         const cudaStream_t stream)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    auto mult_op = thrust::multiplies<float>();
    thrust::transform(exec_policy, gpu_input, gpu_input + range, multiplier_input, products, mult_op);

    auto add_op = thrust::plus<float>();
    *output = thrust::reduce(exec_policy, products, products + range, 0, add_op);
}
*/

__global__ void kernel_multiply_by_frequencies_array(const float* gpu_input,
                                                     float* products_output,
                                                     float* multiplier_input,
                                                     const size_t zx_size,
                                                     const size_t range,
                                                     const size_t full_zx_size,
                                                     const size_t full_range,
                                                     const uint min_depth,
                                                     const size_t buffer_size)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= buffer_size)
        return;

    size_t y = id / zx_size;
    size_t zx_pos = id % zx_size;
    size_t x = zx_pos / range;
    size_t depth = zx_pos % range;

    const size_t full_id = (y * full_zx_size) + (x * full_range) + min_depth + depth;

    float val = gpu_input[full_id];
    if (multiplier_input)
        val *= multiplier_input[depth];
    products_output[id] = val;
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

void multiply_and_reduce(const float* gpu_input,
                      float* gpu_output,
                      float* multiplier_input,
                      const size_t width,
                      const size_t frame_res,
                      const size_t full_range,
                      const uint min_index,
                      const uint max_index,
                      const cudaStream_t stream)
{
    const uint range = max_index - min_index + 1;

    const size_t buffer_size = frame_res * range;

    float* gpu_products = nullptr;
    cudaSafeCall(cudaMalloc(&gpu_products, buffer_size * sizeof(float)));

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(buffer_size, threads);
    kernel_multiply_by_frequencies_array<<<blocks, threads, 0, stream>>>(gpu_input,
                                                                         gpu_products,
                                                                         multiplier_input,
                                                                         width * range,
                                                                         range,
                                                                         width * full_range,
                                                                         full_range,
                                                                         min_index,
                                                                         buffer_size);

    kernel_reduce_block_on_z<<<blocks, threads, 0, stream>>>(gpu_products, gpu_output, frame_res, range);
}

void compute_and_fill_h(const float* gpu_input,
                        float* gpu_output,
                        float* gpu_freq_arr,
                        const size_t width,
                        const size_t frame_res,
                        const size_t full_range,
                        const holovibes::CompositeHSV& hsv_struct,
                        const cudaStream_t stream)
{
    const uint min_h_index = hsv_struct.h.frame_index.min;
    const uint max_h_index = hsv_struct.h.frame_index.max;

    // take z as multiplier (first half of the array)
    float* multiplier_input = gpu_freq_arr;

    float* gpu_h_output = gpu_output;

    multiply_and_reduce(gpu_input,
                     gpu_h_output,
                     multiplier_input,
                     width,
                     frame_res,
                     full_range,
                     min_h_index,
                     max_h_index,
                     stream);
}

void compute_and_fill_s(const float* gpu_input,
                        float* gpu_output,
                        float* gpu_freq_arr,
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

    // take z^2 as multiplier (second half of the array)
    float* multiplier_input = gpu_freq_arr + full_range;

    float* gpu_s_output = gpu_output + (HSV::S * frame_res);

    multiply_and_reduce(gpu_input,
                     gpu_s_output,
                     multiplier_input,
                     width,
                     frame_res,
                     full_range,
                     min_s_index,
                     max_s_index,
                     stream);
}

void compute_and_fill_v(const float* gpu_input,
                        float* gpu_output,
                        float* gpu_freq_arr,
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

    // take z^0 as multiplier (no multiplier)
    float* multiplier_input = nullptr;

    float* gpu_v_output = gpu_output + (HSV::V * frame_res);

    multiply_and_reduce(gpu_input,
                     gpu_v_output,
                     multiplier_input,
                     width,
                     frame_res,
                     full_range,
                     min_v_index,
                     max_v_index,
                     stream);
}

void compute_and_fill_hsv(const float* gpu_input,
                          float* gpu_output,
                          float* gpu_freq_arr,
                          const size_t width,
                          const size_t frame_res,
                          const size_t full_range,
                          const holovibes::CompositeHSV& hsv_struct,
                          const cudaStream_t stream)
{
    compute_and_fill_h(gpu_input, gpu_output, gpu_freq_arr, width, frame_res, full_range, hsv_struct, stream);
    compute_and_fill_s(gpu_input, gpu_output, gpu_freq_arr, width, frame_res, full_range, hsv_struct, stream);
    compute_and_fill_v(gpu_input, gpu_output, gpu_freq_arr, width, frame_res, full_range, hsv_struct, stream);
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

    apply_percentile_and_threshold(gpu_arr,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.h.threshold.min,
                                   hsv_struct.h.threshold.max,
                                   stream);

    map_multiply(gpu_arr, gpu_arr, frame_res, -1.0f, stream);
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
    map_multiply(gpu_arr, gpu_arr, frame_res, 0.66f, stream);
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
    float* gpu_arr_s = gpu_arr + frame_res;

    /*
    apply_percentile_and_threshold(gpu_arr_s,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.s.threshold.min,
                                   hsv_struct.s.threshold.max,
                                   stream);

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_s,
                                                         hsv_struct.s.slider_threshold.min,
                                                         hsv_struct.s.slider_threshold.max,
                                                         frame_res);

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);
    */

    const auto lambda = [] __device__(const float pixel) { return 1.0f; };
    map_generic(gpu_arr_s, gpu_arr_s, frame_res, lambda, stream);
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
    float* gpu_arr_v = gpu_arr + frame_res * 2;

    apply_percentile_and_threshold(gpu_arr_v,
                                   frame_res,
                                   width,
                                   height,
                                   hsv_struct.v.threshold.min,
                                   hsv_struct.v.threshold.max,
                                   stream);

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_v,
                                                         hsv_struct.v.slider_threshold.min,
                                                         hsv_struct.v.slider_threshold.max,
                                                         frame_res);

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);
}

/*
void hsv(const cuComplex* gpu_input,
         float* gpu_output,
         const uint width,
         const uint height,
         const cudaStream_t stream,
         const int time_transformation_size,
         const holovibes::CompositeHSV& hsv_struct)
{
    const uint frame_res = height * width;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    float* gpu_omega_arr = nullptr;
    cudaSafeCall(cudaMalloc(&gpu_omega_arr, sizeof(float) * time_transformation_size * 2)); // w1[] && w2[]

    fill_frequencies_arrays(gpu_omega_arr, frame_res, stream, time_transformation_size);

    float* tmp_hsv_arr;
    cudaSafeCall(cudaMalloc(&tmp_hsv_arr, sizeof(float) * frame_res * 3)); // HSV temp array

    compute_and_fill_hsv(gpu_input, gpu_output, frame_res, gpu_omega_arr, time_transformation_size, hsv_struct, stream);

    kernel_from_interweaved_components_to_distinct_components<<<blocks, threads, 0, stream>>>(gpu_output,
                                                                                              tmp_hsv_arr,
                                                                                              frame_res);
    cudaCheckError();

    // To perform a renormalization, a single min buffer and single max buffer
    // is needed gpu side
    {
        holovibes::cuda_tools::UniquePtr<float> gpu_min(1);
        holovibes::cuda_tools::UniquePtr<float> gpu_max(1);
        apply_operations_on_h(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_s(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_v(tmp_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
    }

    kernel_from_distinct_components_to_interweaved_components<<<blocks, threads, 0, stream>>>(tmp_hsv_arr,
                                                                                              gpu_output,
                                                                                              frame_res);
    cudaCheckError();
    kernel_normalized_convert_hsv_to_rgb<<<blocks, threads, 0, stream>>>(gpu_output, gpu_output, frame_res);
    cudaCheckError();

    map_multiply(gpu_output, gpu_output, frame_res * 3, 65536, stream);

    cudaXFree(tmp_hsv_arr);
    cudaXFree(gpu_omega_arr);
}
*/

__global__ void kernel_rotate_hsv_to_contiguous_z(
    const cuComplex* gpu_input, float* rotated_hsv_arr, const uint frame_res, const uint width, const uint range)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t depth = id / frame_res;
    size_t frame_pos = id % frame_res;
    size_t y = frame_pos / width;
    size_t x = frame_pos % width;
    const size_t rotated_id = (y * range * width) + (x * range) + depth;

    float val = fabsf(gpu_input[id].x);
    rotated_hsv_arr[rotated_id] = val;
}

void rotate_hsv_to_contiguous_z(const cuComplex* gpu_input,
                                float* rotated_hsv_arr,
                                const uint frame_res,
                                const uint width,
                                const uint range,
                                const cudaStream_t stream)
{
    const uint total_size = frame_res * range;
    cudaSafeCall(cudaMalloc(&rotated_hsv_arr, total_size * sizeof(float)));

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(total_size, threads);

    kernel_rotate_hsv_to_contiguous_z<<<blocks, threads, 0, stream>>>(gpu_input,
                                                                      rotated_hsv_arr,
                                                                      frame_res,
                                                                      width,
                                                                      range);
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

    // Frequencies array, used for later operations
    float* gpu_freq_arr = nullptr;
    fill_frequencies_arrays(gpu_freq_arr, frame_res, time_transformation_size, stream);

    // HSV rotated array for contiguous data on z axis
    float* rotated_hsv_arr;
    rotate_hsv_to_contiguous_z(gpu_input, rotated_hsv_arr, time_transformation_size, width, height, stream);

    compute_and_fill_hsv(gpu_input, gpu_output, width, frame_res, gpu_freq_arr, hsv_struct, stream);

    {
        // To perform a renormalization, a single min buffer and single max buffer is needed gpu side
        holovibes::cuda_tools::UniquePtr<float> gpu_min(1);
        holovibes::cuda_tools::UniquePtr<float> gpu_max(1);

        apply_operations_on_h(rotated_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_s(rotated_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
        apply_operations_on_v(rotated_hsv_arr, height, width, gpu_min.get(), gpu_max.get(), hsv_struct, stream);
    }

    kernel_from_distinct_components_to_interweaved_components<<<blocks, threads, 0, stream>>>(tmp_hsv_arr,
                                                                                              gpu_output,
                                                                                              frame_res);
    cudaCheckError();
    kernel_normalized_convert_hsv_to_rgb<<<blocks, threads, 0, stream>>>(gpu_output, gpu_output, frame_res);
    cudaCheckError();

    cudaXFree(rotated_hsv_arr);
    cudaXFree(gpu_freq_arr);
}