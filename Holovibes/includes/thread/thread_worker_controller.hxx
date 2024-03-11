#pragma once


#include <map>

#include "logger.hh"
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
inline void ThreadWorkerController<T>::set_error_callback(std::function<void(const std::exception&)> error_callback)
{
    error_callback_ = error_callback;
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

    LOG_DEBUG("Starting Worker of type {}", typeid(T).name());

    worker_ = std::make_unique<T>(args...);
    thread_ = std::thread(&ThreadWorkerController::run, this);
    Logger::add_thread(thread_.get_id(), typeid(T).name());

    // uint64_t id = std::hash<std::thread::id>{}(thread_.get_id());
    std::stringstream ss;
    ss << thread_.get_id();
    std::string id = ss.str();
    LOG_INFO("Worker of type {} started with ID: {}", typeid(T).name(), id);
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
        LOG_ERROR("Uncaught exception in Worker of type {} : {}", typeid(T).name(), e.what());
        throw;
    }

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);
}
} // namespace holovibes::worker
