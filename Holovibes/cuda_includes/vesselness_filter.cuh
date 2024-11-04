/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "unique_ptr.hh"
#include "common.cuh"
#include "cufft_handle.hh"

using holovibes::cuda_tools::CufftHandle;

void applyConvolutionWithReplicatePadding(const float* image,
                                          float* output,
                                          int imgWidth,
                                          int imgHeight,
                                          const float* kernel,
                                          int kernelWidth,
                                          int kernelHeight);

void normalized_list(float* output, int lim, int size, cudaStream_t stream);

void comp_dgaussian(float* output, float* input, size_t input_size, float sigma, int n, cudaStream_t stream);

void vesselness_filter(float* output,
                       float* input,
                       float sigma,
                       float* g_xx_mul,
                       float* g_xy_mul,
                       float* g_yy_mul,
                       int frame_res,
                       float* convolution_buffer,
                       cuComplex* cuComplex_buffer,
                       CufftHandle* convolution_plan,
                       cublasHandle_t cublas_handler,
                       cudaStream_t stream);