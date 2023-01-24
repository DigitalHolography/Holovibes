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

    if (async_fun_.valid() && worker_ != nullptr)
    {
        LOG_DEBUG("Restarting Worker of type {}", typeid(T).name());
        callback_at_stop_ = [&]()
        {
            LOG_DEBUG("Callback for restarting Worker of type {}", typeid(T).name());
            callback_at_stop_ = []() {};
            worker_ = std::make_unique<T>(args...);
            ThreadWorkerController<T>::run();
        };
        return;
    }

    LOG_DEBUG("Starting Worker of type {}", typeid(T).name());

    worker_ = std::make_unique<T>(args...);
    async_fun_ = std::async(std::launch::async, &ThreadWorkerController::run, this);

    LOG_INFO("Worker of type {} started", typeid(T).name());
}

template <WorkerDerived T>
void ThreadWorkerController<T>::stop()
{
    {
        MutexGuard m_guard(mutex_);

        if (worker_ != nullptr)
        {
            LOG_DEBUG("Request stop of Worker of type {}", typeid(T).name());
            worker_->stop();
        }
    }
}

template <WorkerDerived T>
void ThreadWorkerController<T>::run()
{
    Logger::add_thread(std::this_thread::get_id(), typeid(T).name());

    worker_->run();

    LOG_INFO("Stop worker of type {}", typeid(T).name());

    MutexGuard m_guard(mutex_);
    worker_.reset(nullptr);

    if (callback_at_stop_)
        callback_at_stop_();
}
} // namespace holovibes::worker
