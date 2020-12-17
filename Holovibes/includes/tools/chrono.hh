/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <chrono>

class Chrono
{
  public:
    Chrono() { start(); }

    void start() { start_ = std::chrono::steady_clock::now(); }

    void stop() { end_ = std::chrono::steady_clock::now(); }

    size_t get_seconds()
    {
        return std::chrono::duration_cast<std::chrono::seconds>(end_ - start_)
            .count();
    }

    size_t get_milliseconds()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ -
                                                                     start_)
            .count();
    }

    size_t get_microseconds()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ -
                                                                     start_)
            .count();
    }

    size_t get_nanoseconds()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_ -
                                                                    start_)
            .count();
    }

  private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
    std::chrono::time_point<std::chrono::steady_clock> end_;
};