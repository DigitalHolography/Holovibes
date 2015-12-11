#include "module.hh"

#include <iostream>

namespace holovibes
{
  Module::Module()
    : task_done_(true)
    , stop_requested_(false)
    , thread_(&Module::thread_proc, this)
  {
    cudaStreamCreate(&stream_);
  }

  Module::~Module()
  {
    stop_requested_ = true;

    while (!thread_.joinable())
      continue;
    thread_.join();
  }

  void  Module::push_back_worker(FnType worker)
  {
    workers_.push_back(worker);
  }

  void  Module::push_front_worker(FnType worker)
  {
    workers_.push_front(worker);
  }

  void  Module::thread_proc()
  {
    while (!stop_requested_)
    {
      if (!task_done_)
      {
        for (FnType& w : workers_) w();

        task_done_ = true;
      }
    }
  }
}