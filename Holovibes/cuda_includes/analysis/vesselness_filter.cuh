/*! \file
 *
 * \brief This file transcribe the MatLab code of vesselness_filter in CreateMask of the Pulsewave project for real
 * time.
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

/*! * \brief Applies a vesselness filter to an input image, also known as Frangi's filter.
 *
 * This function applies a vesselness filter to an input image using a series of operations. It involves computing
 * intermediate images, eigenvalues, and applying logical conditions to produce the final output. The function uses
 * the provided CUDA stream for asynchronous execution and a cuBLAS handle for linear algebra operations.
 *
 * \param [out] output Pointer to the output array where the filtered image will be stored.
 * \param [in] input Pointer to the input image array.
 * \param [in] sigma Scalar value used in the computation of the filter.
 * \param [in] g_xx_mul Pointer to the Gaussian kernel array for the xx component.
 * \param [in] g_xy_mul Pointer to the Gaussian kernel array for the xy component.
 * \param [in] g_yy_mul Pointer to the Gaussian kernel array for the yy component.
 * \param [in] kernel_x_size The width of the Gaussian kernel.
 * \param [in] kernel_y_size The height of the Gaussian kernel.
 * \param [in] frame_res The resolution of the input frame (number of elements per frame).
 * \param [in] filter_struct_ Reference to a structure containing intermediate buffers and parameters for the filter.
 * \param [in] cublas_handler The cuBLAS handle to use for linear algebra operations.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function performs the following steps:
 *       1. Computes intermediate images using the `compute_I` function.
 *       2. Copies the intermediate results to a temporary buffer.
 *       3. Computes eigenvalues using the `compute_eigen_values` function.
 *       4. Applies logical conditions and normalization using the `abs_lambda_division`, `normalize`, `If`, and
 * `lambda_2_logical` functions.
 *       5. Uses cuBLAS to find the maximum value in an array.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch and memory operations.
 */
void vesselness_filter(float* output,
                       float* input,
                       float sigma,
                       float* g_xx_mul,
                       float* g_xy_mul,
                       float* g_yy_mul,
                       int kernel_x_size,
                       int kernel_y_size,
                       int frame_res,
                       holovibes::VesselnessFilterEnv& filter_struct_,
                       cublasHandle_t cublas_handler,
                       cudaStream_t stream);

/*!
 * \brief Applies a 2D convolution operation to an input image using a specified kernel.
 *
 * This function applies a 2D convolution operation to an input image using a specified kernel. It configures and
 * launches a CUDA kernel to perform the convolution operation. The function uses the provided CUDA stream for
 * asynchronous execution. The result of the convolution is stored in a temporary buffer and then copied back to the
 * input-output array.
 *
 * \param [in,out] input_output Pointer to the input-output array where the convolution result will be stored.
 * \param [in] kernel Pointer to the convolution kernel array.
 * \param [in] width The width of the input image.
 * \param [in] height The height of the input image.
 * \param [in] kWidth The width of the convolution kernel.
 * \param [in] kHeight The height of the convolution kernel.
 * \param [in] convolution_tmp_buffer Pointer to a temporary buffer used to store the intermediate convolution result.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 * \param [in] padding_type The type of padding to use (e.g., replicate boundary behavior or scalar padding).
 * \param [in] padding_scalar The scalar value to use for padding if `padding_type` is `SCALAR`.
 *
 * \note The function configures the kernel launch parameters based on the image dimensions.
 *       It calls `cudaCheckError()` to check for any CUDA errors after the kernel launch and memory copy operations.
 */
void apply_convolution(float* input_output,
                       const float* kernel,
                       size_t width,
                       size_t height,
                       size_t kWidth,
                       size_t kHeight,
                       float* const convolution_tmp_buffer,
                       cudaStream_t stream,
                       ConvolutionPaddingType padding_type,
                       int padding_scalar = 0);