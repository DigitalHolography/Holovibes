/*! \file
 *
 * \brief Utilities for applying off-axis phase masks.
 */
#pragma once

#include "common.cuh"

/*! \brief Apply a rectangular off-axis phase mask and optional circular shift.
 *
 * \param input[in] Pointer to the input complex frames in frequency domain.
 * \param output[out] Pointer where the masked (and shifted) frames will be written.
 * \param width The frame width.
 * \param height The frame height.
 * \param frame_res The number of pixels in a frame (width * height).
 * \param batch_size Number of frames in the batch.
 * \param x_min Left bound of the retained zone (inclusive).
 * \param x_max Right bound of the retained zone (inclusive).
 * \param y_min Top bound of the retained zone (inclusive).
 * \param y_max Bottom bound of the retained zone (inclusive).
 * \param shift_x Horizontal circular shift applied after masking.
 * \param shift_y Vertical circular shift applied after masking.
 * \param stream CUDA stream used for the operation.
 */
void apply_off_axis_phase_mask_and_shift(const cuComplex* input,
                                         cuComplex* scratch,
                                         cuComplex* destination,
                                         uint width,
                                         uint height,
                                         uint frame_res,
                                         uint batch_size,
                                         int x_min,
                                         int x_max,
                                         int y_min,
                                         int y_max,
                                         int shift_x,
                                         int shift_y,
                                         const cudaStream_t stream);
