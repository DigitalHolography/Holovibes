/*! \file
 *
 * \brief Utils functions for Analysis functions
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"
#include "cublas_handle.hh"

float* load_CSV_to_float_array(const std::string& filename);

void print_in_file_gpu(float* input, uint rows, uint col, std::string filename, cudaStream_t stream);

void print_in_file_cpu(float* input, uint rows, uint col, std::string filename);

void normalized_list(float* output, int lim, int size, cudaStream_t stream);

void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream);

void prepare_hessian(
    float* output, const float* ixx, const float* ixy, const float* iyy, const int size, cudaStream_t stream);

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

void compute_kernel_cuda(float* output, float sigma);