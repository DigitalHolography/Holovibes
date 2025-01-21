/*! \file thread_worker_controller.hh
 *
 * \brief Defines the ThreadWorkerController class. The base class for all worker controllers.
 */
#pragma once

#include "custom_type_traits.hh"
#include "logger.hh"

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
    /*! \brief Default constructor */
    ThreadWorkerController() = default;

    /*! \brief Destructor
     *
     * Stop the thread if it is running
     */
    ~ThreadWorkerController();

    /*! \brief Deleted copy constructor */
    ThreadWorkerController(const ThreadWorkerController<T>&) = delete;

    /*! \brief Deleted copy operator */
    ThreadWorkerController& operator=(const ThreadWorkerController<T>&) = delete;

    /*! \brief Set the function executed at the end of the thread
     *
     * This method must be called before the start method
     */
    void set_callback(std::function<void()> callback);

    /*! \brief Set the function executed when an exception unwind the thread stack
     *
     * This method must be called before the start method
     */
    void set_error_callback(std::function<void(const std::exception&)> error_callback);

    /*! \brief Set the priority of the thread
     *
     * This method must be called before the start method
     *
     * \param priority Priority level of the thread
     */
    void set_priority(int priority);

    /*! \brief Construct the associated worker and start the thread
     *
     * The run method is executed in the thread
     *
     * \param args... The arguments of the constructor of the worker
     */
    template <typename... Args>
    void start(Args&&... args);

    /*! \brief Request the worker to stop. If wait is true, wait for the thread to join.
     *
     * \param[in] wait Whether the function should wait for the worker to stop
     */
    void stop(bool wait = true);

    bool is_running() const { return worker_ != nullptr; }

    /**
     * @brief Update a setting. The actual application of the update
     * might ve delayed until a certain event occurs.
     * @tparam T The type of tho update.
     * @param setting The new value of the setting.
     */
    template <typename S>
    inline void update_setting(S setting)
    {
        if (!worker_)
            return;

        LOG_TRACE("[ThreadWorkerController] [update_setting] {}", typeid(S).name());
        worker_->update_setting(setting);
    }

  private:
    /*! \brief Method run in the thread
     *
     * \details  Call the run method of the associated worker, the callback at the end of the execution and reset the
     * worker
     */
    void run();

    /*! \brief The pointer to the worker that should be controlled */
    std::unique_ptr<T> worker_ = nullptr;
    /*! \brief The thread associated to the worker */
    std::thread thread_;
    /*! \brief The function called at the end of the execution of the thread */
    std::function<void()> callback_ = []() {};
    std::function<void(const std::exception&)> error_callback_ = [](auto) {};
    /*! \brief Mutex used to prevent data races */
    std::mutex mutex_;
};
} // namespace holovibes::worker

#include "thread_worker_controller.hxx"
