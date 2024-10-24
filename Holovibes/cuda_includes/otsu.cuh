/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_desc.hh"

/*! \brief TODO
 *
 * \param frame
 * \param frame_res The total size of a frame (width * height).
 * \param batch_size The size of the batch to transfer.
 * \param stream The CUDA stream on which to launch the operation.
 *
 * \return optimal threshold
 * */
void otsuThreshold(float* frame, const size_t frame_res, const cudaStream_t stream);
void myKernel2_wrapper(float* d_input, float min, float max, const size_t size, const cudaStream_t stream);