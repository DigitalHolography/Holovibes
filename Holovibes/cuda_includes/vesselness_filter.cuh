/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "unique_ptr.hh"
#include "common.cuh"
#include "cufft_handle.hh"

using holovibes::cuda_tools::CufftHandle;

void vesselness_filter(float* output,
                       float* input,
                       float sigma,
                       float* g_xx_mul,
                       float* g_xy_mul,
                       float* g_yy_mul,
                       int kernel_x_size,
                       int kernel_y_size,
                       int frame_res,
                       float* convolution_buffer,
                       cuComplex* cuComplex_buffer,
                       CufftHandle* convolution_plan,
                       cublasHandle_t cublas_handler,
                       cudaStream_t stream);