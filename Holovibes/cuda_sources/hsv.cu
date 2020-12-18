/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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

#define SAMPLING_FREQUENCY 1
/*
 * \brief Convert an array of HSV normalized float to an array of RGB normalized
 * float i.e.: with "[  ]" a pixel: [HSV][HSV][HSV][HSV] -> [RGB][RGB][RGB][RGB]
 * NVdia function
 */

__global__ void kernel_normalized_convert_hsv_to_rgb(const float* src,
                                                     float* dst,
                                                     size_t frame_res)
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

__global__ void kernel_fill_square_frequency_axis(const size_t length,
                                                  float* arr)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length)
    {
        arr[length + id] = arr[id] * arr[id];
    }
}

__global__ void kernel_fill_part_frequency_axis(const size_t min,
                                                const size_t max,
                                                const double step,
                                                const double origin,
                                                float* arr)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (min + id < max)
    {
        arr[min + id] = origin + id * step;
    }
}

void fill_frequencies_arrays(const holovibes::ComputeDescriptor& cd,
                             float* gpu_omega_arr,
                             size_t frame_res,
                             const cudaStream_t stream)
{
    const int time_transformation_size = cd.time_transformation_size;
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    double step = SAMPLING_FREQUENCY / (double)time_transformation_size;
    size_t after_mid_index =
        time_transformation_size / (double)2.0 + (double)1.0;

    kernel_fill_part_frequency_axis<<<blocks, threads, 0, stream>>>(0,
                                                               after_mid_index,
                                                               step,
                                                               0,
                                                               gpu_omega_arr);
    double negative_origin = -SAMPLING_FREQUENCY / (double)2.0;
    negative_origin += time_transformation_size % 2 ? step / (double)2.0 : step;

    kernel_fill_part_frequency_axis<<<blocks, threads, 0, stream>>>(
        after_mid_index,
        time_transformation_size,
        step,
        negative_origin,
        gpu_omega_arr);
    kernel_fill_square_frequency_axis<<<blocks, threads, 0, stream>>>(
        time_transformation_size,
        gpu_omega_arr);
}

/*
** \brief Compute H component of hsv.
*/
__global__ void kernel_compute_and_fill_h(const cuComplex* input,
                                          float* output,
                                          const size_t frame_res,
                                          const size_t min_index,
                                          const size_t max_index,
                                          const size_t total_index,
                                          const size_t omega_size,
                                          const float* omega_arr)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        const size_t index_H = id * 3;
        output[index_H] = 0;
        float summ_p = 0;
        float min = FLT_MAX;

        for (size_t i = min_index; i <= max_index; ++i)
        {
            float input_elm = fabsf(input[i * frame_res + id].x);
            min = fminf(min, input_elm);
            output[index_H] += input_elm * omega_arr[i];
            summ_p += input_elm;
        }

        output[index_H] -= total_index * min;
        output[index_H] /= summ_p;
    }
}

/*
** \brief Compute S component of hsv.
** Could be factorized with H but I kept it like this for the clarity
*/
__global__ void kernel_compute_and_fill_s(const cuComplex* input,
                                          float* output,
                                          const size_t frame_res,
                                          const size_t min_index,
                                          const size_t max_index,
                                          const size_t total_index,
                                          const size_t omega_size,
                                          const float* omega_arr)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        const size_t index_S = id * 3 + 1;
        output[index_S] = 0;

        float summ_p = 0;
        float min = FLT_MAX;

        for (size_t i = min_index; i <= max_index; ++i)
        {
            float input_elm = fabsf(input[i * frame_res + id].x);
            min = fminf(min, input_elm);
            output[index_S] += input_elm * omega_arr[i];
            summ_p += input_elm;
        }

        output[index_S] -= total_index * min;
        output[index_S] /= summ_p;
    }
}

/*
** \brief Compute V component of hsv.
*/
__global__ void kernel_compute_and_fill_v(const cuComplex* input,
                                          float* output,
                                          const size_t frame_res,
                                          const size_t min_index,
                                          const size_t max_index)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        const size_t index_V = id * 3 + 2;
        output[index_V] = 0;
        for (size_t i = min_index; i <= max_index; ++i)
        {
            float input_elm = fabsf(input[i * frame_res + id].x);
            output[index_V] += input_elm;
        }
    }
}

void compute_and_fill_hsv(const cuComplex* gpu_input,
                          float* gpu_output,
                          const size_t frame_res,
                          const holovibes::ComputeDescriptor& cd,
                          float* gpu_omega_arr,
                          size_t omega_arr_size,
                          const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    const uint min_h_index = cd.composite_p_min_h;
    const uint max_h_index = cd.composite_p_max_h;
    const uint min_s_index = cd.composite_p_min_s;
    const uint max_s_index = cd.composite_p_max_s;
    const uint min_v_index = cd.composite_p_min_v;
    const uint max_v_index = cd.composite_p_max_v;

    kernel_compute_and_fill_h<<<blocks, threads, 0, stream>>>(gpu_input,
                                                         gpu_output,
                                                         frame_res,
                                                         min_h_index,
                                                         max_h_index,
                                                         max_h_index -
                                                             min_h_index + 1,
                                                         omega_arr_size,
                                                         gpu_omega_arr);

    if (cd.composite_p_activated_s)
        kernel_compute_and_fill_s<<<blocks, threads, 0, stream>>>(
            gpu_input,
            gpu_output,
            frame_res,
            min_s_index,
            max_s_index,
            max_s_index - min_s_index + 1,
            omega_arr_size,
            gpu_omega_arr + omega_arr_size);
    else
        kernel_compute_and_fill_s<<<blocks, threads, 0, stream>>>(
            gpu_input,
            gpu_output,
            frame_res,
            min_h_index,
            max_h_index,
            max_h_index - min_h_index + 1,
            omega_arr_size,
            gpu_omega_arr + omega_arr_size);

    if (cd.composite_p_activated_v)
        kernel_compute_and_fill_v<<<blocks, threads, 0, stream>>>(gpu_input,
                                                             gpu_output,
                                                             frame_res,
                                                             min_v_index,
                                                             max_v_index);
    else
        kernel_compute_and_fill_v<<<blocks, threads, 0, stream>>>(gpu_input,
                                                             gpu_output,
                                                             frame_res,
                                                             min_h_index,
                                                             max_h_index);

    cudaCheckError();
}

__global__ void threshold_top_bottom(float* output,
                                     const float tmin,
                                     const float tmax,
                                     const uint frame_res)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < frame_res)
    {
        output[id] = fminf(output[id], tmax);
        output[id] = fmaxf(output[id], tmin);
    }
}

__global__ void kernel_from_distinct_components_to_interweaved_components(
    const float* src, float* dst, size_t frame_res)
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

    kernel_from_distinct_components_to_interweaved_components
        <<<blocks, threads, 0, stream>>>(src, dst, frame_res);
}

__global__ void kernel_from_interweaved_components_to_distinct_components(
    const float* src, float* dst, size_t frame_res)
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

    kernel_from_interweaved_components_to_distinct_components<<<blocks,
                                                                threads,
                                                                0,
                                                                stream>>>(src,
                                                                     dst,
                                                                     frame_res);
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
                               percent_in_h,
                               percent_out,
                               2,
                               holovibes::units::RectFd(),
                               false,
                               stream);
    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr,
                                                    percent_out[0],
                                                    percent_out[1],
                                                    frame_res);
}

void apply_gaussian_blur(const holovibes::ComputeDescriptor& cd,
                         float* gpu_arr,
                         uint height,
                         uint width,
                         const cudaStream_t stream)
{
    size_t frame_res = height * width;

    float* gpu_convolution_matrix;
    cudaXMalloc(&gpu_convolution_matrix, frame_res * sizeof(float));
    cudaXMemsetAsync(gpu_convolution_matrix, 0, frame_res * sizeof(float), stream);

    float* blur_matrix = new float[cd.h_blur_kernel_size];
    float blur_value =
        1.0f / (float)(cd.h_blur_kernel_size * cd.h_blur_kernel_size);
    unsigned min_pos_kernel = height / 2 - cd.h_blur_kernel_size / 2;
    for (size_t i = 0; i < cd.h_blur_kernel_size; i++)
    {
        blur_matrix[i] = blur_value;
    }

    // FIXME Might want to replace that with a cudaMemcpy2D
    for (size_t i = 0; i < cd.h_blur_kernel_size; i++)
    {
        cudaXMemcpyAsync(gpu_convolution_matrix + min_pos_kernel +
                        width * (i + min_pos_kernel),
                    blur_matrix,
                    cd.h_blur_kernel_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    }

    shift_corners(gpu_convolution_matrix, 1, width, height, stream);

    cuComplex* gpu_kernel;
    cudaXMalloc(&gpu_kernel, frame_res * sizeof(cuComplex));
    cudaXMemsetAsync(gpu_kernel, 0, frame_res * sizeof(cuComplex), stream);
    cudaSafeCall(cudaMemcpy2DAsync(gpu_kernel,
                              sizeof(cuComplex),
                              gpu_convolution_matrix,
                              sizeof(float),
                              sizeof(float),
                              frame_res,
                              cudaMemcpyDeviceToDevice,
                              stream));

    float* gpu_memory_space;
    cuComplex* gpu_cuComplex_buffer;
    cudaXMalloc(&gpu_memory_space, frame_res * sizeof(float));
    cudaXMalloc(&gpu_cuComplex_buffer, frame_res * sizeof(cuComplex));
    CufftHandle handle{static_cast<int>(width),
                       static_cast<int>(height),
                       CUFFT_C2C};
    convolution_kernel(gpu_arr,
                       gpu_memory_space,
                       gpu_cuComplex_buffer,
                       &handle,
                       width * height,
                       gpu_kernel,
                       false,
                       false,
                       stream);
    cudaCheckError();

    delete[] blur_matrix;
    cudaXFree(gpu_memory_space);
    cudaXFree(gpu_cuComplex_buffer);
    cudaXFree(gpu_convolution_matrix);
    cudaXFree(gpu_kernel);
}

void hsv_normalize(float* const gpu_arr,
                   const uint frame_res,
                   float* const gpu_min,
                   float* const gpu_max,
                   const cudaStream_t stream)
{
    reduce_min(gpu_arr, gpu_min, frame_res, stream); // Get the minimum value
    reduce_max(gpu_arr, gpu_max, frame_res, stream); // Get the maximum value

    const auto lambda = [gpu_min, gpu_max] __device__(const float pixel) {
        return (pixel - *gpu_min) * (1 / (*gpu_max - *gpu_min));
    };
    map_generic(gpu_arr, gpu_arr, frame_res, lambda, stream);
}

void apply_operations_on_h(const holovibes::ComputeDescriptor& cd,
                           float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const cudaStream_t stream)
{
    const uint frame_res = height * width;
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    apply_percentile_and_threshold(gpu_arr,
                                   frame_res,
                                   width,
                                   height,
                                   cd.composite_low_h_threshold,
                                   cd.composite_high_h_threshold,
                                   stream);

    map_multiply(gpu_arr, gpu_arr, frame_res, -1.0f, stream);
    hsv_normalize(gpu_arr, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr,
                                                    cd.slider_h_threshold_min,
                                                    cd.slider_h_threshold_max,
                                                    frame_res);
    if (cd.h_blur_activated)
    {
        apply_gaussian_blur(cd, gpu_arr, height, width, stream);
    }

    hsv_normalize(gpu_arr, frame_res, gpu_min, gpu_max, stream);
    map_multiply(gpu_arr, gpu_arr, frame_res, 0.66f, stream);
}

void apply_operations_on_s(const holovibes::ComputeDescriptor& cd,
                           float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
                           const cudaStream_t stream)
{
    const uint frame_res = height * width;
    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);
    float* gpu_arr_s = gpu_arr + frame_res;

    apply_percentile_and_threshold(gpu_arr_s,
                                   frame_res,
                                   width,
                                   height,
                                   cd.composite_low_s_threshold,
                                   cd.composite_high_s_threshold,
                                   stream);

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_s,
                                                    cd.slider_s_threshold_min,
                                                    cd.slider_s_threshold_max,
                                                    frame_res);

    hsv_normalize(gpu_arr_s, frame_res, gpu_min, gpu_max, stream);
}

void apply_operations_on_v(const holovibes::ComputeDescriptor& cd,
                           float* gpu_arr,
                           uint height,
                           uint width,
                           float* const gpu_min,
                           float* const gpu_max,
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
                                   cd.composite_low_v_threshold,
                                   cd.composite_high_v_threshold,
                                   stream);

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);

    threshold_top_bottom<<<blocks, threads, 0, stream>>>(gpu_arr_v,
                                                    cd.slider_v_threshold_min,
                                                    cd.slider_v_threshold_max,
                                                    frame_res);

    hsv_normalize(gpu_arr_v, frame_res, gpu_min, gpu_max, stream);
}

void hsv(const cuComplex* gpu_input,
         float* gpu_output,
         const uint width,
         const uint height,
         const holovibes::ComputeDescriptor& cd,
         const cudaStream_t stream)
{
    const int time_transformation_size = cd.time_transformation_size;
    const uint frame_res = height * width;

    const uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(frame_res, threads);

    float* gpu_omega_arr = nullptr;
    cudaXMalloc(&gpu_omega_arr,
                sizeof(float) * time_transformation_size * 2); // w1[] && w2[]

    fill_frequencies_arrays(cd, gpu_omega_arr, frame_res, stream);

    float* tmp_hsv_arr;
    cudaXMalloc(&tmp_hsv_arr, sizeof(float) * frame_res * 3); // HSV temp array

    compute_and_fill_hsv(gpu_input,
                         gpu_output,
                         frame_res,
                         cd,
                         gpu_omega_arr,
                         time_transformation_size,
                         stream);

    kernel_from_interweaved_components_to_distinct_components<<<blocks,
                                                                threads,
                                                                0,
                                                                stream>>>(
        gpu_output,
        tmp_hsv_arr,
        frame_res);
    cudaCheckError();

    // To perform a renormalization, a single min buffer and single max buffer
    // is needed gpu side
    {
        holovibes::cuda_tools::UniquePtr<float> gpu_min(1);
        holovibes::cuda_tools::UniquePtr<float> gpu_max(1);
        apply_operations_on_h(cd,
                              tmp_hsv_arr,
                              height,
                              width,
                              gpu_min.get(),
                              gpu_max.get(),
                              stream);
        apply_operations_on_s(cd,
                              tmp_hsv_arr,
                              height,
                              width,
                              gpu_min.get(),
                              gpu_max.get(),
                              stream);
        apply_operations_on_v(cd,
                              tmp_hsv_arr,
                              height,
                              width,
                              gpu_min.get(),
                              gpu_max.get(),
                              stream);
    }

    kernel_from_distinct_components_to_interweaved_components<<<blocks,
                                                                threads,
                                                                0,
                                                                stream>>>(
        tmp_hsv_arr,
        gpu_output,
        frame_res);
    cudaCheckError();
    kernel_normalized_convert_hsv_to_rgb<<<blocks, threads, 0, stream>>>(gpu_output,
                                                                    gpu_output,
                                                                    frame_res);
    cudaCheckError();

    map_multiply(gpu_output, gpu_output, frame_res * 3, 65536, stream);

    cudaXFree(tmp_hsv_arr);
    cudaXFree(gpu_omega_arr);
}
