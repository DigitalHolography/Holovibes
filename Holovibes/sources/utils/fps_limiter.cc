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
    auto target_frame_time = std::chrono::duration<double>(1.0 / (double)target_fps);
    auto end_time = last_time_called_ + target_frame_time;

    // std::this_thread::sleep_until(end_time); TODO: should be better but breaks some tests 
    while(std::chrono::high_resolution_clock::now() < end_time) {} //ugly but works  

    last_time_called_ = std::chrono::high_resolution_clock::now();
}
} // namespace holovibes