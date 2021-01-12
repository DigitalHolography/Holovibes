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
// Fast forward declaration
class Worker;

template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;

/*!
 *  \brief    Class used to control the execution of a worker inside a thread
 *
 *  \details  T must derived from the Worker class
 */
template <Derived<Worker> T>
class ThreadWorkerController
{
  public:
    /*!
     *  \brief    Default constructor
     */
    ThreadWorkerController() = default;

    /*!
     *  \brief    Destructor
     *
     *  \details  Stop the thread if it is running
     */
    ~ThreadWorkerController();

    /*!
     *  \brief    Deleted copy constructor
     */
    ThreadWorkerController(const ThreadWorkerController<T>&) = delete;

    /*!
     *  \brief    Deleted copy operator
     */
    ThreadWorkerController&
    operator=(const ThreadWorkerController<T>&) = delete;

    /*!
     *  \brief    Set the function executed at the end of the thread
     *
     *  \details  This method must be called before the start method
     */
    void set_callback(std::function<void()> callback);

    /*!
     *  \brief    Construct the associated worker and start the thread
     *
     *  \details  The run method is executed in the thread
     *
     *  \param    args...   The arguments of the constructor of the worker
     */
    template <typename... Args>
    void start(Args&&... args);

    /*!
     *  \brief    Join the thread
     *
     *  \details  If the worker was running, stop it before joining the
     *            associated thread
     */
    void stop();

  private:
    /*!
     *  \brief    Method run in the thread
     *
     *  \details  Call the run method of the associated worker,
     *            the callback at the end of the execution and reset the worker
     */
    void run();

    //! The pointer to the worker that should be controlled
    std::unique_ptr<T> worker_ = nullptr;
    //! The thread associated to the worker
    std::thread thread_;
    //! The function called at the end of the execution of the thread
    std::function<void()> callback_ = []() {};
    //! Mutex used to prevent data races
    std::mutex mutex_;
};
} // namespace holovibes::worker

#include "thread_worker_controller.hxx"