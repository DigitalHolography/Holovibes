#pragma once

#include "rect.hh"

/*!
 * \brief Compute the output percentile of the xy view
 *
 * No offset needed
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
 * \brief Compute the output percentile of the xz view
 *
 * \param offset Number of rows to skip from the top and bottom of the view
 */
void compute_percentile_xz_view(const float* gpu_input,
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
 * \param offset Number of columns to skip from the left and right of the view
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
