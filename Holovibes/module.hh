#pragma once

# include <thread>

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
    Module(bool *finish);

    //!< Join the thread before exiting.
    ~Module();

    //!< Add an extra task to carry at each iteration.
    void  add_worker(FnType worker);

    //!< The function used by the created thread.
    void  thread_proc();

  private:
    //!< Pointer to the boolean managing activity/waiting.
    bool*       finish_;
    //!< Set to true by the Module to stop and join its thread, when asked to stop.
    bool        stop_requested_;
    //!< All tasks are carried out sequentially, in growing index order.
    FnVector    workers_;
    std::thread thread_;
  };
}