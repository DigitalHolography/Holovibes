/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

void prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size, cudaStream_t stream);

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream);

void apply_diaphragm_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream);
                       
void print_in_file(float* input, uint size, std::string filename, cudaStream_t stream);

void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream);