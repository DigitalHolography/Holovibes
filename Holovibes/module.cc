#include "module.hh"

#include <iostream>

namespace holovibes
{
  Module::Module(bool *finish)
    : finish_(finish), stop_requested_(false)
    , thread_(&Module::thread_proc, this)
  {}

  Module::~Module()
  {
    stop_requested_ = true;

    while (!thread_.joinable())
      continue;
    thread_.join();
  }

  void  Module::add_worker(FnType worker)
  {
    workers_.push_back(worker);
  }

  void  Module::thread_proc()
  {
    while (!stop_requested_)
    {
      while (!stop_requested_ && *finish_)
        continue;

      for (FnType& w : workers_) w();

      *finish_ = true;
    }
  }
}