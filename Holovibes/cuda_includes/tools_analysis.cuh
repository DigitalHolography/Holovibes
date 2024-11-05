/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream);

void load_kernel_in_GPU(cuComplex* output, const float* kernel, const size_t frame_res, cudaStream_t stream);

void convolution_kernel_add_padding(float* output,
                                    float* kernel,
                                    const int width,
                                    const int height,
                                    const int new_width,
                                    const int new_height,
                                    cudaStream_t stream);

__global__ void kernel_4D_eigenvalues(float* H, float* lambda_1, float* lambda_2, int rows, int cols);

void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename);

void print_in_file(float* input, uint size, std::string filename, cudaStream_t stream);