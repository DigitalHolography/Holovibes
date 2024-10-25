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

void convolution_float(float* output,
                       const float* input1,
                       const float* input2,
                       const uint size,
                       const cufftHandle plan2d_a,
                       const cufftHandle plan2d_b,
                       const cufftHandle plan2d_inverse,
                       cudaStream_t stream);

// void xcorr2(float* output, float* input1, float* input2, const short width, const short height, cudaStream_t stream);

void xcorr2(float* output,
            float* input1,
            float* input2,
            const short width,
            const short height,
            cudaStream_t stream,
            cufftComplex* d_freq_1,
            cufftComplex* d_freq_2,
            cufftComplex* d_corr_freq);
// cufftHandle plan_2d,
// cufftHandle plan_2dinv);