/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

/*! \brief  Divide all the pixels of input image(s) by the float divider.
 *
 * \param image The image(s) to process. Should be contiguous memory.
 * \param frame_res Size of the frame
 * \param divider Divider value for all elements.
 * \param batch_size The number of images
 */
__global__ void
kernel_complex_divide(cuComplex* image, const uint frame_res, const float divider, const uint batch_size);

/*! \brief  Multiply the pixels value of 2 complexe input images
 *
 * The images to multiply should have the same size.
 * The result is given in output.
 * Output should have the same size of inputs.
 */
__global__ void
kernel_multiply_frames_complex(const cuComplex* input1, const cuComplex* input2, cuComplex* output, const uint size);

/*! \brief  divide pixels value of numerator float input images by denominator
 *
 * The images to divide should have the same size.
 * The result is given in output.
 * Output should have the same size of inputs.
 */
__global__ void
kernel_divide_frames_float(const float* numerator, const float* denominator, float* output, const uint size);

/*! \brief  Multiply the pixels value of 2 complexe input images
 *
 * The images to multiply should have the same size.
 * The result is given in output.
 * Output should have the same size of inputs.
 */
void multiply_frames_complex(
    const cuComplex* input1, const cuComplex* input2, cuComplex* output, const uint size, const cudaStream_t stream);

/*! \brief normalize input according to a renormalize constant
 *
 * \param input input data
 * \param result_reduce device double pointer used to store the result of the
 * reduction operation required to compute mean value of the frame
 * \param frame_res frame resolution
 * \param norm_constant Constant of the normalization
 */
void gpu_normalize(float* const input,
                   double* const result_reduce,
                   const size_t frame_res,
                   const uint norm_constant,
                   const cudaStream_t stream);