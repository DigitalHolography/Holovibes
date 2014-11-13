#include "stdafx.h"

#include "thread_compute.hh"
#include "pipeline.hh"
#include <cassert>

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    const ComputeDescriptor& desc,
    Queue& input_q)
    : compute_desc_(desc)
    , input_q_(input_q)
    , output_q_(nullptr)
    , compute_on_(true)
    , thread_(&ThreadCompute::thread_proc, this)
  {
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();

    delete output_q_;
  }

  Queue& ThreadCompute::get_queue()
  {
    return *output_q_;
  }

  void ThreadCompute::thread_proc()
  {
    camera::FrameDescriptor fd = input_q_.get_frame_desc();
    fd.depth = 2;

    output_q_ = new Queue(fd, input_q_.get_max_elts());
    assert(output_q_);
    output_q_->dequeue();
#if 0
    Queue& output = *output_q_;
    output.print();
#endif
    std::cout << compute_desc_.nsamples << std::endl;
    Pipeline p(input_q_, *output_q_, compute_desc_);
    while (compute_on_)
      p.exec();
  }
}