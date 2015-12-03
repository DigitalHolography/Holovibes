#pragma once

# include <vector>
#include <cuda_runtime.h>
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

    virtual ~Pipeline();

    virtual void exec() override;

  protected:
    virtual void update_n_parameter(unsigned short n) override;

    virtual void refresh() override;

    virtual void record_float();

  private:
    void step_forward();

  private:
    std::vector<Module*>        modules_;
    std::vector<cudaStream_t>   streams_;

    std::vector<float*>         gpu_float_buffers_;
    std::vector<cufftComplex*>  gpu_complex_buffers_;
    short*                      gpu_short_buffer_;

    bool*                       is_finished_;
  };
}