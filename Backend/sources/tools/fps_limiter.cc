#include "fps_limiter.hh"

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
    if (target_fps == 0)
    {
        target_fps = 1; // Minimum FPS to avoid division by zero
    }
    chrono_.start();
    chrono_.wait(1.0 / (double)target_fps);
}
} // namespace holovibes
