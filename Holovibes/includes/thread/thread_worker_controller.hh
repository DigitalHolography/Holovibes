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

        ThreadWorkerController& operator=(const ThreadWorkerController<T>&) = delete;

        void set_callback(std::function<void()> callback);

        template <typename... Args>
        void start(Args&&... args);

        void request_stop();

        void stop();

    private:
        void run();

        std::unique_ptr<T> worker_ = nullptr;

        std::thread thread_;

        std::function<void()> callback_ = [](){};

        std::mutex mutex_;
    };
} // namespace holovibes::worker

#include "thread_worker_controller.hxx"