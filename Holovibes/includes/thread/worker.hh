/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

namespace holovibes::worker
{
/*!
 *  \brief    Abstract class that represents a worker doing a task
 */
class Worker
{
  public:
    /*!
     *  \brief    Default constructor
     */
    Worker() = default;

    /*!
     *  \brief    Stop the execution of the worker
     */
    virtual void stop();

    /*!
     *  \brief    Core method of the worker, main method of its execution
     */
    virtual void run() = 0;

  protected:
    //! Whether the worker should stop its execution
    std::atomic<bool> stop_requested_{false};
};
} // namespace holovibes::worker