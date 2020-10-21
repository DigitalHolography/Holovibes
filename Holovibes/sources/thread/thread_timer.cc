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
        auto manager = gui::InfoManager::get_manager();
        auto start = std::chrono::high_resolution_clock::now();
        while (!stop_)
        {
            auto end = std::chrono::high_resolution_clock::now();

            // if one second has passed
            if (std::chrono::duration_cast<std::chrono::seconds>(end - start).count() >= 1)
            {
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