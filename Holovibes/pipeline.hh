#pragma once

# include <vector>
# include <cuda_runtime.h>
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

    void stop_pipeline();

    virtual void exec() override;

  protected:
    virtual void update_n_parameter(unsigned short n) override;

    virtual void refresh() override;

    virtual void record_float() override;

    template <class T>
    Module* create_module(std::vector<T*>& gpu_buffers, size_t buf_size);
  private:
    void step_forward();

  private:
    //!< All Modules regrouping all tasks to be carried out, in order.
    std::vector<Module*>        modules_;
    /*! Each Module needs to be bound to a stream at initialization.
     * Hence, the stream cannot be stored in the Module class. */
    std::vector<cudaStream_t>   streams_;

    //!< Working sets of 'nsamples' frames of complex data.
    std::vector<float*>         gpu_float_buffers_;
    //!< Working sets of a single frame (the p'th Fourier component) of float data.
    std::vector<cufftComplex*>  gpu_complex_buffers_;
    // A single frame containing 16-bit pixel values, used for display.
    short*                      gpu_short_buffer_;

    /*! A table the same size as modules_.size(). Each Module indicates here wether
     * its current task is done or not, so the Pipeline can manage everyone. */
    std::vector<bool*>           is_finished_;
    bool                        stop_threads_;
  };
}