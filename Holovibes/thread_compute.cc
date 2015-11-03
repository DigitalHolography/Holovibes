#include "thread_compute.hh"
#include "pipeline.hh"
#include <cassert>

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    ComputeDescriptor& desc,
    Queue& input,
    Queue& output,
    bool is_float_output_enabled,
    std::string float_output_file_src,
    unsigned int float_output_nb_frame)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , pipeline_(nullptr)
    , compute_on_(true)
    , memory_cv_()
    , is_float_output_enabled_(is_float_output_enabled)
    , thread_(&ThreadCompute::thread_proc, this, float_output_file_src, float_output_nb_frame)
  {
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCompute::thread_proc(
    std::string float_output_file_src,
    unsigned int float_output_nb_frame)
  {
    pipeline_ = std::shared_ptr<Pipeline>(new Pipeline(input_, output_, compute_desc_));

    if (is_float_output_enabled_)
    {
      pipeline_->request_float_output(float_output_file_src, float_output_nb_frame);
      pipeline_->refresh();
    }

    memory_cv_.notify_one();

    while (compute_on_)
      pipeline_->exec();
  }
}