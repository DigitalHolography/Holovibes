#pragma once

# include <thread>

# include "pipeline_utils.hh"

namespace holovibes
{
  class Module
  {
  public:
    Module(bool *finish);

    ~Module();
    void  add_worker(FnType worker);

    void  thread_proc();

  private:
    bool*       finish_;
    bool        stop_requested_;
    FnVector    workers_;
    std::thread thread_;
  };
}