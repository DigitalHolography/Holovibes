/*! \file
 *
 * \brief Utils functions for Analysis functions
 */
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef unsigned int uint;

void normalized_list(float* output, int lim, int size, cudaStream_t stream);

void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream);

void prepare_hessian(float* output, const float* I, const int size, const size_t offset, cudaStream_t stream);

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream);

void apply_diaphragm_mask(float* output,
                          const float center_X,
                          const float center_Y,
                          const float radius,
                          const short width,
                          const short height,
                          const cudaStream_t stream);

void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream);

void compute_circle_mask(float* output,
                         const float center_X,
                         const float center_Y,
                         const float radius,
                         const short width,
                         const short height,
                         const cudaStream_t stream);
void apply_mask_and(
    float* output, const float* input, const short width, const short height, const cudaStream_t stream);

void apply_mask_or(float* output, const float* input, const short width, const short height, const cudaStream_t stream);

float* compute_gauss_deriviatives_kernel(
    int kernel_width, int kernel_height, float sigma, cublasHandle_t cublas_handler_, cudaStream_t stream);

void convolution_kernel_add_padding(float* output,
                                    float* kernel,
                                    const int width,
                                    const int height,
                                    const int new_width,
                                    const int new_height,
                                    cudaStream_t stream);

float* compute_kernel(float sigma);

void compute_gauss_kernel(float* output, float sigma, cudaStream_t stream);

int count_non_zero(const float* const input, const int rows, const int cols, cudaStream_t stream);

void divide_frames_inplace(float* const input_output,
                           const float* const denominator,
                           const uint size,
                           cudaStream_t stream);

void normalize_array(float* device_array, size_t size, float min_range, float max_range, cudaStream_t stream);