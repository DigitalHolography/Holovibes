#include <algorithm>

#include "pipeline.hh"

namespace holovibes
{
  Pipeline::Pipeline(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : ICompute(input, output, desc)
    , is_finished_ { nullptr }
  {
    // TODO : Initialize modules by binding resources to std::functions.
    //        Allocate is_finished_ and set every value to false.
  }

  Pipeline::~Pipeline()
  {
    std::for_each(modules_.begin(),
      modules_.end(),
      [](Module* module) { delete module; });
    std::for_each(streams_.begin(),
      streams_.end(),
      [](cudaStream_t& stream) { cudaStreamDestroy(stream); });

    std::for_each(gpu_float_buffers_.begin(),
      gpu_float_buffers_.end(),
      [](float* buffer) { cudaFree(buffer); });
    std::for_each(gpu_complex_buffers_.begin(),
      gpu_complex_buffers_.end(),
      [](cufftComplex* buffer) { cudaFree(buffer); });
    cudaFree(gpu_short_buffer_);

    delete[] is_finished_;
  }

  void Pipeline::exec()
  {
    // TODO
  }

  void Pipeline::update_n_parameter(unsigned short n)
  {
    ICompute::update_n_parameter(n);
    // TODO
  }

  void Pipeline::refresh()
  {
    // TODO
  }

  void Pipeline::step_forward()
  {
    std::rotate(gpu_float_buffers_.begin(),
      gpu_float_buffers_.begin() + 1,
      gpu_float_buffers_.end());
    std::rotate(gpu_complex_buffers_.begin(),
      gpu_complex_buffers_.begin() + 1,
      gpu_complex_buffers_.end());
  }

  void Pipeline::record_float()
  {
    // TODO
  }
}