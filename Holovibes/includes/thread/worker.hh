/*! \file
 *
 * \brief Defines the abstract Worker class for managing tasks.
 */
#pragma once

#include <atomic>

/*! \brief Namespace for worker-related classes in the Holovibes project */
namespace holovibes::worker
{
/*! \class Worker
 *
 * \brief Abstract class that represents a worker executing a task.
 *
 * This class provides the basic interface and functionality for a worker, including 
 * methods to start and stop the worker's execution.
 */
class Worker
{
  public:
    /*! \brief Default constructor */
    Worker() = default;

    /*! \brief Virtual destructor to ensure proper cleanup in derived classes */
    virtual ~Worker() = default;

    /*! \brief Stops the execution of the worker */
    virtual void stop();

    /*! \brief Core method of the worker, representing the main execution loop.
     *
     * This method must be implemented by derived classes.
     */
    virtual void run() = 0;

  protected:
    /*! \brief Indicates whether the worker should stop its execution */
    std::atomic<bool> stop_requested_{false};
};
} // namespace holovibes::worker