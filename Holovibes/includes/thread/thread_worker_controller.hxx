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
    MutexGuard m_guard(mutex_);

    LOG_DEBUG("Starting Worker of type {}", typeid(T).name());

    worker_ = std::make_unique<T>(args...);
    async_fun_ = std::async(std::launch::async, &ThreadWorkerController::run, this);

    LOG_INFO("Worker of type {} started", typeid(T).name());
}

template <WorkerDerived T>
void ThreadWorkerController<T>::stop()
{
    LOG_DEBUG("Call stop of Worker of type {}", typeid(T).name());
    {
        MutexGuard m_guard(mutex_);

        if (worker_ != nullptr)
        {
            LOG_DEBUG("Request stop of Worker of type {}", typeid(T).name());
            worker_->stop();
        }
        else
        {
            LOG_WARN("Will NOT stop of Worker of type {}", typeid(T).name());
        }
    }
}

template <WorkerDerived T>
void ThreadWorkerController<T>::run()
{
    Logger::add_thread(std::this_thread::get_id(), typeid(T).name());

    try
    {
        worker_->run();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Exception in Worker {} of type {}", typeid(T).name(), e.what());
    }

    LOG_INFO("Stop worker of type {}", typeid(T).name());

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);
}
} // namespace holovibes::worker
