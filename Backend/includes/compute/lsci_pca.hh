#pragma once

#include <memory>
#include <vector>
#include "frame_desc.hh"

namespace holovibes::compute
{
struct LsciPcaPatchResult
{
    unsigned x, y, width, height;
    unsigned static_components = 0;
    std::vector<double> eigenvalues;
};

struct LsciPcaResult
{
    std::vector<float> original;
    std::vector<float> static_component;
    std::vector<float> dynamic;
    std::vector<double> eigenvalues;
    unsigned static_components = 0;
    // Local mode has one spectrum/component count per patch, rather than a
    // single global spectrum. Coordinates are zero-based input-image pixels.
    std::vector<LsciPcaPatchResult> patches;
};

// Offline, real-intensity PCA followed by spatiotemporal LSCI. No live setting
// or time-transformation enum exposes this processor to the GUI.
class LsciPca
{
  public:
    // pca_patch_size = 0 uses the whole image; 2..512 selects disjoint square
    // PCA regions clipped at the image edges. The LSCI window is independent.
    LsciPca(unsigned width, unsigned height, unsigned frames = 100, unsigned spatial_window = 7,
            unsigned pca_patch_size = 0);
    ~LsciPca();
    LsciPca(const LsciPca&) = delete;
    LsciPca& operator=(const LsciPca&) = delete;

    void add_frame(const void* host_frame, camera::PixelDepth depth);
    LsciPcaResult process();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
