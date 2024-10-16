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
 * \param input_output Buffer on which the convolution will be applied
 * \param convolved_buffer Buffer used for convolution calcul (will be overwriten)
 * \param cuComplex_buffer Used for complex computation
 * \param plan Plan2D used for the three fft
 * \param size The size of the image (it is a square)
 * \param gpu_kernel Array of float which is the convolution's kernel
 * \param divide_convolution_enabled Activate the division of the input by the convolved image
 * \param stream The operation stream
 */
void convolution_kernel(float* input_output,
                        float* gpu_convolved_buffer,
                        cuComplex* cuComplex_buffer,
                        CufftHandle* plan,
                        const size_t size,
                        const cuComplex* gpu_kernel,
                        const bool divide_convolution_enabled,
                        const cudaStream_t stream);
