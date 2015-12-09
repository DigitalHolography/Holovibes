#pragma once

# include <vector>
# include <list>
# include <cuda_runtime.h>
# include <cufft.h>

# include "icompute.hh"
# include "pipeline_utils.hh"
# include "module.hh"
# include "queue.hh"
# include "compute_descriptor.hh"

namespace holovibes
{
  /*! The Pipeline is a parallel computing model,
   * grouping tasks in parallel modules.
   *
   * Whereas the Pipe executes sequentially its operations
   * at each iteration, the Pipeline handles a set of Modules,
   * each containing some defined tasks. The Pipeline orders the
   * Modules to work synchronously on independent data sets,
   * moving at each iteration the target data set of each Module.
   * This model allows for enhanced performances but consumes vast
   * amounts of memory on the GPU; its usage is limited to calculations
   * on small block sizes (nsamples). */
  class Pipeline : public ICompute
  {
  public:
    Pipeline(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);

    //!< Stop the Modules and free all resources.
    virtual ~Pipeline();

    //!< Stop Modules, clear them, and free resources.
    void stop_pipeline();

    //!< The computing loop.
    virtual void exec() override;

  private:
    //!< Clear the contents of the Pipeline and fetch new computations to use.
    virtual void refresh() override;

    virtual void record_float() override;

    /*! \brief Realloc all buffer with the new nsamples and update ICompute */
    virtual void update_n_parameter(unsigned short n);

    /*! Create a module, allocate its data set, and create its CUDA stream.
     * \param gpu_buffers The container in which the data set.
     * \param buf_size The number of elements of type T to allocate. */
    template <class T>
    Module* create_module(std::list<T*>& gpu_buffers, size_t buf_size);

    //!< For each Module, advance the dataset being worked on.
    void step_forward();

  private:
    //!< All Modules regrouping all tasks to be carried out, in order.
    std::vector<Module*>        modules_;

    /*! \brief Number of step before exec call refresh
    * \note Implemented for auto contrast, it need pipeline initialized */
    unsigned int step_count_before_refresh_;

    //!< Working sets of 'nsamples' frames of complex data.
    std::vector<cufftComplex*>  gpu_complex_buffers_;
    //!< Working sets of a single frame (the p'th Fourier component) of complex data.
    std::vector<cufftComplex*>  gpu_pindex_buffers_;
    //!< Working sets of a single frame (the p'th Fourier component) of float data.
    std::vector<float*>         gpu_float_buffers_;
    //!< A single frame containing 16-bit pixel values, used for display.
    unsigned short              *gpu_short_buffer_;
  };
}