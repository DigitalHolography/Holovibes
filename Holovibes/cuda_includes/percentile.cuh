/*! \file
 *
 * \brief Declaration of compute percentile functions
 */
#pragma once

#include "rect.hh"

/*!
 * \brief Compute the output percentile of the xy view
 *
 *  No offset needed
 *
 * \param[in] gpu_input The input image
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] offset Number of columns to skip from the left and right of the view
 * \param[in] h_percent The percentile to compute
 * \param[out] h_out_percent The output percentile
 * \param[in] size_percent The size of the percentile array
 * \param[in] sub_zone The zone to apply the mask to
 * \param[in] compute_on_sub_zone Whether to compute the percentile on the sub zone
 * \param[in] stream The CUDA stream to use
 */
void compute_percentile_xy_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const holovibes::units::RectFd& sub_zone,
                                const bool compute_on_sub_zone,
                                const cudaStream_t stream);

/*!
 * \brief Compute the output percentile of the yz view
 *
 * \param[in] gpu_input The input image
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] offset Number of columns to skip from the left and right of the view
 * \param[in] h_percent The percentile to compute
 * \param[out] h_out_percent The output percentile
 * \param[in] size_percent The size of the percentile array
 * \param[in] sub_zone The zone to apply the mask to
 * \param[in] compute_on_sub_zone Whether to compute the percentile on the sub zone
 * \param[in] stream The CUDA stream to use
 */
void compute_percentile_yz_view(const float* gpu_input,
                                const uint width,
                                const uint height,
                                uint offset,
                                const float* const h_percent,
                                float* const h_out_percent,
                                const uint size_percent,
                                const holovibes::units::RectFd& sub_zone,
                                const bool compute_on_sub_zone,
                                const cudaStream_t stream);
