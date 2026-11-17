/*! \file
 * \brief Spatially local PCA clutter filtering of one complex temporal block.
 */
#pragma once

#include <memory>
#include <cuComplex.h>
#include <cuda_runtime_api.h>

namespace holovibes::compute
{
/*! \brief Remove the strongest effective-rank components independently in each spatial tile.
 *
 * Input and output are column-major (width * height) x frames complex matrices,
 * exactly like the existing PCA time transform. Tile origins are (i * stride_x,
 * j * stride_y), strictly inside the image; samples beyond either edge wrap to
 * the opposite edge. The output averages residual complex samples at each
 * original pixel/time coordinate, before the usual magnitude/phase conversion.
 * Temporal block scheduling belongs to the compute pipe, not this processor.
 */
class LocalPca
{
  public:
    LocalPca(unsigned width,
             unsigned height,
             unsigned frames,
             unsigned width_x,
             unsigned width_y,
             unsigned stride_x,
             unsigned stride_y,
             cudaStream_t stream);
    ~LocalPca();
    LocalPca(const LocalPca&) = delete;
    LocalPca& operator=(const LocalPca&) = delete;

    /*! \brief Filter a complete block using ceil(rank_factor * effective_rank) removed components.
     * Input and output must be distinct device buffers. A zero factor copies the
     * input unchanged; the removal count is capped at the temporal dimension.
     */
    void process(const cuComplex* input, cuComplex* output, float rank_factor = 1.f);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace holovibes::compute
