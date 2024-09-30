#pragma once

/**
 * @file fps_limiter.hh
 * This file provides a fps limiter class to use in a loop
 * in order to control the number of "loop pass" per seconds.
 */

#include "chrono.hh"
#include <cstddef>

namespace holovibes
{
/**
 * @brief The FPS limiter can be used to limit the number of
 * "pass" in a loop.
 */
class FPSLimiter
{
  public:
    /**
     * @brief Construct a new FPSLimiter object.
     */
    FPSLimiter();

    /**
     * @brief Wait until the target FPS is reached. One should call this function
     * in a loop to control its speed.
     * @param target_fps The targeted fps.
     */
    void wait(size_t target_fps);

  private:
    Chrono chrono_;
};
} // namespace holovibes
