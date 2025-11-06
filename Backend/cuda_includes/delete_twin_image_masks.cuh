/*! \file
 *
 * \brief Helpers to build masks used by the delete twin image space transformation.
 */
#pragma once

#include "common.cuh"

namespace holovibes
{
namespace cuda
{

/*!
 * \brief Fill the Mp and Ma masks for the delete twin image transform.
 *
 * Mp is set to 1 inside the main rectangle and 0 elsewhere.
 * Ma is set to 0 inside the main rectangle and its central symmetry, 1 elsewhere.
 *
 * \param[out] mp_mask Pointer to the mask dedicated to the phase (Mp).
 * \param[out] ma_mask Pointer to the mask dedicated to the amplitude (Ma).
 * \param[in] width Frame width.
 * \param[in] height Frame height.
 * \param[in] rect_x_min Left bound of the main rectangle (inclusive).
 * \param[in] rect_x_max Right bound of the main rectangle (exclusive).
 * \param[in] rect_y_min Top bound of the main rectangle (inclusive).
 * \param[in] rect_y_max Bottom bound of the main rectangle (exclusive).
 * \param[in] sym_x_min Left bound of the symmetric rectangle (inclusive).
 * \param[in] sym_x_max Right bound of the symmetric rectangle (exclusive).
 * \param[in] sym_y_min Top bound of the symmetric rectangle (inclusive).
 * \param[in] sym_y_max Bottom bound of the symmetric rectangle (exclusive).
 * \param[in] stream CUDA stream used to launch the kernel.
 */
void build_delete_twin_image_masks(float* mp_mask,
                                   float* ma_mask,
                                   int width,
                                   int height,
                                   int rect_x_min,
                                   int rect_x_max,
                                   int rect_y_min,
                                   int rect_y_max,
                                   int sym_x_min,
                                   int sym_x_max,
                                   int sym_y_min,
                                   int sym_y_max,
                                   const cudaStream_t stream);

} // namespace cuda
} // namespace holovibes
