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

#include "common.cuh"
#include "chart_point.hh"

/*! \brief  Make the sum of input and selected zone
*
* \param height The height of the input image.
* \param width The width of the input image.
* \param zone Coordinates of the zone to use.
*/
void apply_zone_sum(const float *input,
			  const uint height,
			  const uint width,
			  double *output,
			  const holovibes::units::RectFd& zone,
			  const cudaStream_t stream = 0);

/*! \brief  Make the std sum ( sum of (x_i - x_avg) ** 2 for i in [1, N] ) of input and selected zone
*
* \param height The height of the input image.
* \param width The width of the input image.
* \param zone Coordinates of the zone to use.
*/
void apply_zone_std_sum(const float *input,
	const uint height,
	const uint width,
	double *output,
	const holovibes::units::RectFd& zone,
	const double avg_signal,
	const cudaStream_t stream = 0);

/*! \brief  Make the chart plot on the 2 selected zones
*
* \param width The width of the input image.
* \param height The height of the input image.
* \param signal_zone Coordinates of the signal zone to use.
* \param noise_zone Coordinates of the noise zone to use.
* \return ChartPoint containing all computations for one point of chart
*/
holovibes::ChartPoint make_chart_plot(float* input,
						const uint width,
						const uint height,
						const holovibes::units::RectFd&	signal_zone,
						const holovibes::units::RectFd&	noise_zone,
						const cudaStream_t stream = 0);