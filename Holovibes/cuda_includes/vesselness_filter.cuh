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

float* comp_dgaussian(float* x, size_t x_size, float sigma, int n);

float* vesselness_filter(float* input,
                         float sigma,
                         float* g_xx_mul,
                         float* g_xy_mul,
                         float* g_yy_mul,
                         size_t kernel_height,
                         size_t kernel_width,
                         int frame_size,
                         float* convolution_buffer,
                         cuComplex* cuComplex_buffer,
                         CufftHandle* convolution_plan,
                         cudaStream_t stream);
