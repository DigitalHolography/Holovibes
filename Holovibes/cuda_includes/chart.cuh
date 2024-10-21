/*!
 * \file chart.cuh
 *
 * \brief Contains functions for making chart plots on two selected zones using CUDA.
 *
 * Usage:
 * - To compute the chart plot on two regions of interest in an image, use `make_chart_plot`.
 * - The function takes an input buffer, image dimensions, signal and noise zones, and a CUDA stream.
 * - It returns a `ChartPoint` struct containing the computed signal, noise, and derived metrics.
 *
 * Code Example:
 * #include "chart_plot.hh"
 * #include <cuda_runtime.h>
 *
 * // Image dimensions
 * const uint width = 1920;
 * const uint height = 1080;
 *
 * // Allocate and initialize input buffer
 * float* input = buffers_.gpu_postprocess_frame;
 *
 * // Define signal and noise zones
 * auto signal_zone = setting<settings::SignalZone>();
 * auto noise_zone = setting<settings::NoiseZone>();
 *
 * // Create CUDA stream
 * cudaStream_t stream;
 * cudaStreamCreate(&stream);
 *
 * // Compute chart point
 * holovibes::ChartPoint chart_point = make_chart_plot(input, width, height, signal_zone, noise_zone, stream);
 *
 * // Use the results
 * // ...
 *
 * // Clean up
 * cudaStreamDestroy(stream);
 */

#pragma once

#include "common.cuh"
#include "chart_point.hh"

/*! \brief Make the chart plot on the 2 selected zones
 *
 * \param[in] input The input buffer containing image data
 * \param[in] width The width of the input image
 * \param[in] height The height of the input image
 * \param[in] signal_zone The region of interest in the image where the signal is measured
 * \param[in] noise_zone The region of interest in the image where the noise is measured
 * \param[in] stream The CUDA stream on which the computations will be launched
 * \return A ChartPoint struct containing the computed signal, noise, and their derived metrics
 */
holovibes::ChartPoint make_chart_plot(float* __restrict__ input,
                                      const uint width,
                                      const uint height,
                                      const holovibes::units::RectFd& signal_zone,
                                      const holovibes::units::RectFd& noise_zone,
                                      const cudaStream_t stream);
