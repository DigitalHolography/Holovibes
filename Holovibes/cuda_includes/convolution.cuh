/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "cufft_handle.hh"
using holovibes::cuda_tools::CufftHandle;

/*! \brief This function allows us to apply a convolution (with a kernel) to frames
 *
 * \param input Buffer on which the convolution will be applied
 * \param convolved_buffer Buffer used for convolution calcul (will be overwriten)
 * \param plan Plan2D used for the three fft
 * \param frame_width Width of the frame
 * \param frame_height Height of the frame
 * \param kernel Array of float which is the convolution's kernel
 * \param divide_convolution_enabled Activate the division of the input by the convolved image
 */
void convolution_kernel(float* gpu_input,
                        float* gpu_convolved_buffer,
                        cuComplex* cuComplex_buffer,
                        CufftHandle* plan,
                        const size_t size,
                        const cuComplex* gpu_kernel,
                        const bool divide_convolution_enabled,
                        const bool normalize_enabled,
                        const cudaStream_t stream);

/*! \brief Computes the 2D cross-correlation of 2 images.
 *  The computations are optimized in frequency domain using the following formula :
 *  XCORR(input1, input2)=FFT-1(FFT(input1)⋅conj(FFT(input2)))
 *  The cross-correlation matrix is stored in an output buffer and the buffers and plans used in
 *  frequency domain need to be given in parameter for optimization purpose to avoid multiple
 *  allocations.
 *  \param[out] output The output buffer of the cross-correlation matrix, should be the size of inputs
 *  images.
 *  \param[in] input1 The first image.
 *  \param[in] input2 The second image.
 *  \param[in] d_freq_output The buffer for the output matrix in frequency domain (already allocated).
 *  \param[in] d_freq_1 The buffer for the first image in frequency domain (already allocated).
 *  \param[in] d_freq_2 The buffer for the second image in frequency domain (already allocated).
 *  \param[in] plan_2d The R2C plan to go in frequency domain. CUFFT_R2C 2D plan
 *  \param[in] plan_2dinv The C2R plan to go in real domain. CUFFT_C2R 2D plan
 *  \param[in] freq_size The size of frequency buffers, computed from: width * (height / 2 + 1)
 *  \param[in] stream The CUDA stream for the kernels.
 */
void xcorr2(float* output,
            float* input1,
            float* input2,
            cufftComplex* d_freq_output,
            cufftComplex* d_freq_1,
            cufftComplex* d_freq_2,
            cufftHandle plan_2d,
            cufftHandle plan_2dinv,
            const int freq_size,
            cudaStream_t stream);