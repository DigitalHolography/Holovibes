/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "unique_ptr.hh"
#include "common.cuh"
#include "cufft_handle.hh"

using holovibes::cuda_tools::CufftHandle;

float* comp_dgaussian(float* x, float sigma, int n, int x_size);

float* gaussian_imfilter_sep(float* input_img,
                             float* kernel,
                             const size_t frame_res,
                             float* convolution_buffer,
                             cuComplex* cuComplex_buffer,
                             CufftHandle& convolution_plan,
                             cudaStream_t stream);

float* vesselness_filter(float* input,
                         float sigma,
                         float* g_xx_mul,
                         float* g_xy_mul,
                         float* g_yy_mul,
                         int frame_size,
                         float* convolution_buffer,
                         cuComplex* cuComplex_buffer,
                         CufftHandle* convolution_plan,
                         cudaStream_t stream);
