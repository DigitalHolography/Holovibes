/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */
#pragma once

#include "rect.hh"

/*!
* \brief Compute the output percentile of the xz view
*
* \param offset Number of rows to skip from the top and bottom of the view
*/
void compute_percentile_xz_view(const float *gpu_input,
							   const uint width,
							   const uint height,
							   uint offset,
							   const float* const h_percent,
							   float* const h_out_percent,
							   const uint size_percent,
							   const holovibes::units::RectFd& sub_zone,
							   const bool compute_on_sub_zone);

/*!
* \brief Compute the output percentile of the yz view
*
* \param offset Number of columns to skip from the left and right of the view
*/
void compute_percentile_yz_view(const float *gpu_input,
							   const uint width,
							   const uint height,
							   uint offset,
							   const float* const h_percent,
							   float* const h_out_percent,
							   const uint size_percent,
							   const holovibes::units::RectFd& sub_zone,
							   const bool compute_on_sub_zone);

/*!
* \brief Compute the output percentile of the xy view
*
* No offset needed
*/
void compute_percentile_xy_view(const float *gpu_input,
								const uint width,
								const uint height,
								const float* const h_percent,
								float* const h_out_percent,
								const uint size_percent,
								const holovibes::units::RectFd& sub_zone,
								const bool compute_on_sub_zone);