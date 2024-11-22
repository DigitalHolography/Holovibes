/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "compute_env.hh"

using holovibes::cuda_tools::CufftHandle;

enum ConvolutionPaddingType
{
    REPLICATE = 0,
    SCALAR,
};

void vesselness_filter(float* output,
                       float* input,
                       float sigma,
                       float* g_xx_mul,
                       float* g_xy_mul,
                       float* g_yy_mul,
                       int kernel_x_size,
                       int kernel_y_size,
                       int frame_res,
                       holovibes::VesselnessFilterStruct& filter_struct_,
                       cublasHandle_t cublas_handler,
                       cudaStream_t stream);

void apply_convolution(float* input_output,
                       const float* kernel,
                       size_t width,
                       size_t height,
                       size_t kWidth,
                       size_t kHeight,
                       cudaStream_t stream,
                       ConvolutionPaddingType padding_type,
                       int padding_scalar = 0);