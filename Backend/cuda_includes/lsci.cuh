/*! \file
 * \brief Pooled spatiotemporal laser speckle contrast on direct camera intensities.
 */
#pragma once

#include "common.cuh"
#include "frame_desc.hh"

/*! \brief Accumulate sum(I) and sum(I^2) for a consecutive batch of raw intensity frames.
 *
 * Input frames are contiguous and are read as unsigned 8-bit, unsigned 16-bit, or
 * floating-point 32-bit intensities. No display scaling or complex modulus is applied.
 * Set reset on the first batch of each nonoverlapping temporal block; otherwise the
 * existing moments are continued. All operations use the supplied CUDA stream.
 */
void lsci_accumulate(float2* moments,
                     const void* input,
                     size_t frame_res,
                     unsigned batch_size,
                     camera::PixelDepth depth,
                     bool reset,
                     cudaStream_t stream);

/*! \brief Produce K = population standard deviation / mean over a clipped spatial neighborhood.
 *
 * Each moments entry contains the sum and squared sum over frame_count frames.
 * Spatial and temporal samples are pooled into one ensemble. Negative computed
 * variances are clamped to zero; nonpositive local means produce zero. The float
 * output is dimensionless and must not alias moments. spatial_window is the odd
 * side length in pixels, from 1 through 63; the default is the paper's 5x5 window.
 * Invalid spatial window sizes throw std::invalid_argument.
 */
void lsci_contrast(float* output,
                   const float2* moments,
                   unsigned width,
                   unsigned height,
                   unsigned frame_count,
                   cudaStream_t stream,
                   unsigned spatial_window = 5);
