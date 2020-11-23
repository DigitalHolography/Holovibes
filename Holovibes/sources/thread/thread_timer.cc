#include "thread_timer.hh"

#include "MainWindow.hh"

#include <chrono>
#include <atomic>

namespace holovibes
{
    ThreadTimer::ThreadTimer(std::atomic<uint>& nb_frame_one_second)
        : thread(&ThreadTimer::run, this),
          nb_frame_one_second_(nb_frame_one_second),
          stop_(false)
    {}

    void ThreadTimer::run()
    {
        /* To avoid calling get_manager in the while loop, we get the manager here
        * Calling get_manager with no gui will cause a runtime execption
        */
        gui::InfoManager* manager = nullptr;
        if (!gui::InfoManager::is_cli())
            manager = gui::InfoManager::get_manager();

        auto start = std::chrono::high_resolution_clock::now();
        while (!stop_)
        {
            auto end = std::chrono::high_resolution_clock::now();

            // if one second has passed
            if (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() >= 1)
            {
                if (!gui::InfoManager::is_cli())
                    manager->insert_info(gui::InfoManager::InfoType::INPUT_FPS, "InputFps", std::to_string(nb_frame_one_second_) + std::string(" fps"));
                nb_frame_one_second_ = 0;
                start = std::chrono::high_resolution_clock::now();
            }
        }
    }

    ThreadTimer::~ThreadTimer()
    {
        stop();
        while (!joinable())
            continue;
        join();
    }
}