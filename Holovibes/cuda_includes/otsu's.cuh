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
 * \param depth The pixel depth.
 * \param stream The CUDA stream on which to launch the operation.
 *
 * \return optimal threshold
 * */
int otsuThreshold(float* frame,
                  const size_t frame_res,
                  const int batch_size,
                  const camera::PixelDepth depth,
                  const cudaStream_t stream);