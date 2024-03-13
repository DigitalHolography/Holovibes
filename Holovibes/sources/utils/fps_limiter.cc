#include "utils/fps_limiter.hh"

#include <chrono>
#include <thread>
#include <cstddef>
#include <spdlog/spdlog.h>

namespace holovibes
{
FPSLimiter::FPSLimiter()
    : last_time_called_(std::chrono::high_resolution_clock::now())
{
}

void FPSLimiter::wait(size_t target_fps)
{
    auto target_frame_time = std::chrono::duration<double>(1.0 / static_cast<double>(target_fps));
    auto end_time = last_time_called_ + target_frame_time;

    auto now = std::chrono::high_resolution_clock::now();
    if (now < end_time)
        std::this_thread::sleep_for(end_time - now);

    last_time_called_ = std::move(end_time);
}
} // namespace holovibes