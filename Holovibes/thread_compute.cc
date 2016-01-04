#include "thread_compute.hh"
#include "pipe.hh"
#include "pipeline.hh"
#include <cassert>

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    ComputeDescriptor& desc,
    Queue& input,
    Queue& output,
    const PipeType pipetype)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , pipetype_(pipetype)
    , pipe_(nullptr)
    , memory_cv_()
    , thread_(&ThreadCompute::thread_proc, this)
  {
  }

  ThreadCompute::~ThreadCompute()
  {
    pipe_->request_termination();

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCompute::thread_proc()
  {
    if (pipetype_ == PipeType::PIPE)
      pipe_ = std::shared_ptr<ICompute>(new Pipe(input_, output_, compute_desc_));
    else
      pipe_ = std::shared_ptr<ICompute>(new Pipeline(input_, output_, compute_desc_));

    memory_cv_.notify_one();

    pipe_->exec();
  }
}