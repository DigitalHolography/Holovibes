#include "utils/fps_limiter.hh"

#include <chrono>
#include <thread>
#include <cstddef>
#include <spdlog/spdlog.h>
namespace holovibes
{
FPSLimiter::FPSLimiter()
    : chrono_()
{
}

void FPSLimiter::wait(size_t target_fps)
{
    chrono_.start();
    chrono_.wait(1.0f / (double)target_fps);
}

void FPSLimiter::wait_for(double seconds)
{
    chrono_.start();
    chrono_.wait(seconds);
}
} // namespace holovibes
