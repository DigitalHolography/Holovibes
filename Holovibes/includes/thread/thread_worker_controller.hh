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
class Worker;

template <class T, class U>
concept Derived = std::is_base_of<U, T>::value;

template <Derived<Worker> T>
class ThreadWorkerController
{
  public:
    ThreadWorkerController() = default;

    ~ThreadWorkerController();

    ThreadWorkerController(const ThreadWorkerController<T>&) = delete;

    ThreadWorkerController&
    operator=(const ThreadWorkerController<T>&) = delete;

    void set_callback(std::function<void()> callback);

    /*!
     *  \brief    Set the priority of the thread
     *
     * \param priority Priority level of the thread
     *
     *  \details  This method must be called before the start method
     */
    void set_priority(int priority);

    template <typename... Args>
    void start(Args&&... args);

    void stop();

  private:
    void run();

    std::unique_ptr<T> worker_ = nullptr;

    std::thread thread_;

    std::function<void()> callback_ = []() {};

    std::mutex mutex_;
};
} // namespace holovibes::worker

#include "thread_worker_controller.hxx"