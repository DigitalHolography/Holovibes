/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"
#include "chart_point.hh"

/*! \brief  Make the chart plot on the 2 selected zones
 *
 * \param width The width of the input image.
 * \param height The height of the input image.
 * \param signal_zone Coordinates of the signal zone to use.
 * \param noise_zone Coordinates of the noise zone to use.
 * \return ChartPoint containing all computations for one point of chart
 */
holovibes::ChartPoint make_chart_plot(float* __restrict__ input,
                                      const uint width,
                                      const uint height,
                                      const holovibes::units::RectFd& signal_zone,
                                      const holovibes::units::RectFd& noise_zone,
                                      const cudaStream_t stream);
