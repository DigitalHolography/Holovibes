#include "utils/fps_limiter.hh"

#include <chrono>
#include <thread>
#include <cstddef>
#include <spdlog/spdlog.h>

namespace holovibes
{
FPSLimiter::FPSLimiter()
    : chrono_()
    , between_()
{
    between_.start();
}

void FPSLimiter::wait(size_t target_fps)
{
    ancient_between_ += (double)between_.get_seconds();
    between_.start();
    if (ancient_between_ > (1.0 / (double)target_fps))
    {
        ancient_between_ -= (1.0 / (double)target_fps);
    }
    else
    {
        chrono_.start();
        chrono_.wait((1.0 / (double)target_fps) - ancient_between_);
        ancient_between_ = 0;
    }
}
} // namespace holovibes
