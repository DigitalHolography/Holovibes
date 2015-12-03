#include "thread_compute.hh"
#include "pipe.hh"
#include <cassert>

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    ComputeDescriptor& desc,
    Queue& input,
    Queue& output,
    bool is_float_output_enabled,
    const std::string float_output_file_src,
    const unsigned int float_output_nb_frame)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , pipe_(nullptr)
    , memory_cv_()
    , is_float_output_enabled_(is_float_output_enabled)
    , thread_(&ThreadCompute::thread_proc, this, float_output_file_src, float_output_nb_frame)
  {
  }

  ThreadCompute::~ThreadCompute()
  {
    pipe_->request_termination();

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCompute::thread_proc(std::string float_output_file_src,
    const unsigned int float_output_nb_frame)
  {
    pipe_ = std::shared_ptr<ICompute>(new Pipe(input_, output_, compute_desc_));

    if (is_float_output_enabled_)
    {
      pipe_->request_float_output(float_output_file_src, float_output_nb_frame);
      pipe_->refresh();
    }

    memory_cv_.notify_one();

    pipe_->exec();
  }
}