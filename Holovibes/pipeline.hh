#pragma once

# include <vector>
# include <cufft.h>

# include "icompute.hh"

# include "pipeline_utils.hh"
# include "module.hh"
# include "queue.hh"
# include "compute_descriptor.hh"

namespace holovibes
{
  class Pipeline : public ICompute
  {
  public:
    Pipeline(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);

  private:
    std::vector<Module*>        modules_;
    bool*                       is_finished_;
    bool                        termination_requested_;
    std::vector<float*>         gpu_float_buffers_;
    std::vector<cufftComplex*>  gpu_complex_buffers_;
    short*                      gpu_short_buffer_;
  };
}