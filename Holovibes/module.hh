#pragma once

# include <thread>
# include <cuda_runtime.h>

# include "pipeline_utils.hh"

namespace holovibes
{
  /*! Regroups one or several predefined tasks,
   * which shall work on a unique data buffer.
   *
   * A module uses a thread to work on these tasks.
   * A Module needs to be handled by a upper level manager,
   * without which it will not synchronize with other Modules. */
  class Module
  {
  public:
    /*! Initialize a module with no tasks, and the address
     to a boolean value managing its activity. */
    Module();

    //!< Join the thread before exiting.
    ~Module();

    //!< Add an extra task after other to carry at each iteration.
    void  push_back_worker(FnType worker);
    //!< Add an extra task before other task to carry at each iteration.
    void  push_front_worker(FnType worker);

    //!< The function used by the created thread.
    void  thread_proc();

  public:
    //!< Boolean managing activity/waiting.
    bool          task_done_;
    //!< Each Modules need stream given to their worker
    cudaStream_t  stream_;
  private:
    //!< Set to true by the Module to stop and join its thread, when asked to stop.
    bool        stop_requested_;
    //!< All tasks are carried out sequentially, in growing index order.
    FnDeque     workers_;
    std::thread thread_;
  };
}