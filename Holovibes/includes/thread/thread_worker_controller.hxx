/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
    inline void ThreadWorkerController<T>::set_callback(std::function<void()> callback)
    {
        callback_ = callback;
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
                worker_->stop();;
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