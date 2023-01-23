/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <future>

namespace holovibes::worker
{
// Fast forward declaration
class Worker;

template <class T>
concept WorkerDerived = std::is_base_of<Worker, T>::value;

/*! \class ThreadWorkerController
 *
 * \brief Class used to control the execution of a worker inside a thread
 *
 * \details T must derived from the Worker class
 */
template <WorkerDerived T>
class ThreadWorkerController
{
  public:
    ThreadWorkerController() {}

    /*! \brief Destructor
     *
     * Stop the thread if it is running
     */
    ~ThreadWorkerController() { stop(); }

    /*! \brief Deleted copy constructor */
    ThreadWorkerController(const ThreadWorkerController<T>&) = delete;

    /*! \brief Deleted copy operator */
    ThreadWorkerController& operator=(const ThreadWorkerController<T>&) = delete;

    /*! \brief Construct the associated worker and start the thread
     *
     * The run method is executed in the thread
     *
     * \param args... The arguments of the constructor of the worker
     */
    template <typename... Args>
    void start(Args&&... args);

    /*! \brief Join the thread
     *
     * If the worker was running, stop it before joining the associated thread
     */
    void stop();

    bool is_running() const { return worker_ != nullptr; }
    void join() { async_fun_.wait(); }

  private:
    /*! \brief Method run in the thread
     *
     * \details  Call the run method of the associated worker, the callback at the end of the execution and reset
     * the worker
     */
    void run();

    /*! \brief The pointer to the worker that should be controlled */
    std::unique_ptr<T> worker_ = nullptr;

    std::future<void> async_fun_;

    /*! \brief Mutex used to prevent data races */
    std::mutex mutex_;

    std::function<void(void)> callback_at_stop_;
};
} // namespace holovibes::worker

#include "thread_worker_controller.hxx"
