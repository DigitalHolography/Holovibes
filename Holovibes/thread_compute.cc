#include "stdafx.h"

#include "thread_compute.hh"
#include "pipeline.hh"
#include <cassert>

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    ComputeDescriptor& desc,
    Queue& input,
    Queue& output)
    : compute_desc_(desc)
    , input_(input)
    , output_(output)
    , pipeline_(nullptr)
    , compute_on_(true)
    , thread_(&ThreadCompute::thread_proc, this)
  {}

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCompute::thread_proc()
  {
    pipeline_ = new Pipeline(input_, output_, compute_desc_);

    while (compute_on_)
      pipeline_->exec();

    delete pipeline_;
  }
}