/*! \file
 *  \brief Functions used to parallelize the computation used for the stabilization.
 *  The stabilization performs computation as follows:
 *  - Apply a circular mask to each images where each values outside of the circle is 0 and each
 *  values inside the circle is 1.
 *  - Compute the mean inside the circle for each image and substract it to rescale the data.
 *  - Apply a cross-correlation between an image choose as reference and
 *  each image of the buffer.
 *  - Take the argmax of the result of the cross-correlation, representing the point to stabilize
 *  which is the center of the eye.
 *  - Shifts all the images to stabilize them to the reference.
 *
 *  For now the process is planned to take up to 4 CUDA kernels but feel free to optimize if possible.
 *
 *  The reference is the mean of the last 3 images of the buffer, to have a sliding window through the
 *  time.
 */
#pragma once

#include "common.cuh"
#include "cuda_runtime.h"
#include "tools.hh"
#include "cuda_memory.cuh"
#include "apply_mask.cuh"
#include "masks.cuh"

using uint = unsigned int;

/*! \brief Apply the first step of the process by getting a circular mask using `kernel_circular_mask`.
 *  Computes the center of the image and a circular mask to keep only the center of the
 *  eye.
 *
 *  \param[out] output The output image after mask application.
 *  \param[in] input The gpu_post_process_buffer on which the mask is applied.
 *  \param[in out] pixels_mean Pointer to store the mean of the pixels inside the circle.
 *  \param[in] width The width of an image.
 *  \param[in] height The height of an image.
 *  \param[in] stream The CUDA stream on which to launch the operation.
 */
void stabilization_get_mask(float* mask, short width, short height, const cudaStream_t stream);