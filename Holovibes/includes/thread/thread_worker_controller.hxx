#pragma once

#include <map>

#include "logger.hh"
#include "thread_worker_controller.hh"

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

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

    LOG_INFO("Worker of type {} started with ID: {}", typeid(T).name(), thread_.get_id());
}

template <WorkerDerived T>
void ThreadWorkerController<T>::stop()
{
    {
        MutexGuard m_guard(mutex_);

        if (worker_ != nullptr)
            worker_->stop();
    }

    if (thread_.joinable() && thread_.get_id() != std::this_thread::get_id())
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
        error_callback_(e);
        LOG_ERROR("Uncaught exception in Worker of type {} : {}", typeid(T).name(), e.what());
        throw;
    }

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);
}
} // namespace holovibes::worker
