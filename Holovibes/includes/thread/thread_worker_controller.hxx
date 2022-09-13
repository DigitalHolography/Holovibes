#pragma once

#include "thread_worker_controller.hh"

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

template <WorkerDerived T>
ThreadWorkerController<T>::~ThreadWorkerController()
{
    stop();
}

template <WorkerDerived T>
inline void ThreadWorkerController<T>::set_callback(std::function<void()> callback)
{
    callback_ = callback;
}

template <WorkerDerived T>
inline void ThreadWorkerController<T>::set_priority(int priority)
{
    SetThreadPriority(thread_.native_handle(), priority);
}

template <WorkerDerived T>
template <typename... Args>
void ThreadWorkerController<T>::start(Args&&... args)
{
    stop();

    MutexGuard m_guard(mutex_);

    // LOG_TRACE << "Starting Worker of type " << typeid(T).name();

    worker_ = std::make_unique<T>(args...);
    thread_ = std::thread(&ThreadWorkerController::run, this);

    // // LOG_TRACE << "Worker of type " << typeid(T).name() << " started with ID: " << thread_.get_id();
}

template <WorkerDerived T>
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

template <WorkerDerived T>
void ThreadWorkerController<T>::run()
{
    try
    {
        worker_->run();
        callback_();
    }
    catch (const std::exception& e)
    {
        // LOG_ERROR << "Uncaught exception in Worker of type " << typeid(T).name() << " : " << e.what();
        throw;
    }

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);
}
} // namespace holovibes::worker
