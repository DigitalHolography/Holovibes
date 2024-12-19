/*! \file
 *
 * \brief declaration of the Worker class
 */
#pragma once

/*!
 * \namespace holovibes::worker
 *
 * \brief Namespace containing the Worker class
 */
namespace holovibes::worker
{
/*! \class Worker
 *
 * \brief Abstract class that represents a worker doing a task
 */
class Worker
{
  public:
    /*! \brief Default constructor */
    Worker() = default;

    /*! \brief Stop the execution of the worker */
    virtual void stop();

    /*! \brief Core method of the worker, main method of its execution */
    virtual void run() = 0;

  protected:
    /*! \brief Whether the worker should stop its execution */
    std::atomic<bool> stop_requested_{false};
};
} // namespace holovibes::worker
