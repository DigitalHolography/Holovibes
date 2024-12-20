/*! \file
 *
 * \brief Declaration functions use for chart computation
 */
#pragma once

#include "common.cuh"
#include "chart_point.hh"

/*!
 * \brief Make the sum of input and selected zone
 *
 * \param[in] input The input image
 * \param[in] height The height of the input image
 * \param[in] width The width of the input image
 * \param[out] output The output image
 * \param[in] zone The zone to apply the mask to
 * \param[in] mask The mask to apply
 * \param[in] stream The CUDA stream to use
 */
void apply_zone_sum(const float* input,
                    const uint height,
                    const uint width,
                    double* output,
                    const holovibes::units::RectFd& zone,
                    const cudaStream_t stream);

/*! \brief  Make the std sum ( sum of (x_i - x_avg) ** 2 for i in [1, N] ) of input and selected zone
 *
 * \param[in] input The input image
 * \param[in] height The height of the input image
 * \param[in] width The width of the input image
 * \param[out] output The output image
 * \param[in] zone The zone to apply the mask to
 * \param[in] avg_signal The average signal
 * \param[in] stream The CUDA stream to use
 */
void apply_zone_std_sum(const float* input,
                        const uint height,
                        const uint width,
                        double* output,
                        const holovibes::units::RectFd& zone,
                        const double avg_signal,
                        const cudaStream_t stream);

/*! \brief  Make the chart plot on the 2 selected zones
 *
 * \param[in] input The input image
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] signal_zone The zone to apply the mask to
 * \param[in] noise_zone The zone to apply the mask to
 * \param[in] stream The CUDA stream to use
 *
 * \return ChartPoint containing all computations for one point of chart
 */
holovibes::ChartPoint make_chart_plot(float* input,
                                      const uint width,
                                      const uint height,
                                      const holovibes::units::RectFd& signal_zone,
                                      const holovibes::units::RectFd& noise_zone,
                                      const cudaStream_t stream);
