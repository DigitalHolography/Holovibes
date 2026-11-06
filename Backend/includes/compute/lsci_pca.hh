#pragma once

#include <memory>
#include <vector>
#include "frame_desc.hh"

namespace holovibes::compute
{
struct LsciPcaResult
{
    std::vector<float> original;
    std::vector<float> static_component;
    std::vector<float> dynamic;
    std::vector<double> eigenvalues;
    unsigned static_components = 0;
};

// Offline, real-intensity PCA followed by spatiotemporal LSCI. No live setting
// or time-transformation enum exposes this processor to the GUI.
class LsciPca
{
  public:
    LsciPca(unsigned width, unsigned height, unsigned frames = 100, unsigned spatial_window = 7);
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
