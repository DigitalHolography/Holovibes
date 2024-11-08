/*! \file
 *
 * \brief Utils functions for Analysis functions
 */
#pragma once

#include "cuda_memory.cuh"
#include "common.cuh"

float* load_CSV_to_float_array(const std::string& filename);

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

void compute_circle_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream);
void apply_mask_and(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream);
void apply_mask_or(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream);