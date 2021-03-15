/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "thread_worker_controller.hh"

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

template <Derived<Worker> T>
ThreadWorkerController<T>::~ThreadWorkerController()
{
    stop();
}

template <Derived<Worker> T>
inline void
ThreadWorkerController<T>::set_callback(std::function<void()> callback)
{
    callback_ = callback;
}

template <Derived<Worker> T>
inline void ThreadWorkerController<T>::set_priority(int priority)
{
    SetThreadPriority(thread_.native_handle(), priority);
}

template <Derived<Worker> T>
template <typename... Args>
void ThreadWorkerController<T>::start(Args&&... args)
{
    stop();

    MutexGuard m_guard(mutex_);

    worker_ = std::make_unique<T>(args...);
    thread_ = std::thread(&ThreadWorkerController::run, this);
}

template <Derived<Worker> T>
void ThreadWorkerController<T>::stop()
{
    {
        MutexGuard m_guard(mutex_);

        if (worker_ != nullptr)
            worker_->stop();
    }

    if (thread_.joinable())
        thread_.join();
}

template <Derived<Worker> T>
void ThreadWorkerController<T>::run()
{
    worker_->run();
    callback_();

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);
}
} // namespace holovibes::worker