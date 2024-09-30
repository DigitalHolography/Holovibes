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
    chrono_.wait(1.0 / (double)target_fps);
}
} // namespace holovibes