/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

__global__ void prepareHessian(float* output, const float* ixx, const float* ixy, const float* iyx, const float* iyy, const int size);

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream);

void convolution_kernel_add_padding(float* output,
                                    float* kernel,
                                    const int width,
                                    const int height,
                                    const int new_width,
                                    const int new_height,
                                    cudaStream_t stream);

void compute_sorted_eigenvalues_2x2(float* H, int frame_res, float* lambda1, float* lambda2, cudaStream_t stream);

void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename);

void print_in_file(float* input, uint size, std::string filename, cudaStream_t stream);

void calculerValeursPropres(float a, float b, float d, float* lambda1, float* lambda2, int ind);