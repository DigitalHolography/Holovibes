/*! \file
 *
 * \brief Wrapper around the chrono library
 */
#pragma once

#include <chrono>

/*! \class Chrono
 *
 * \brief #TODO Add a description for this class
 */
class Chrono
{
  public:
    Chrono() { start(); }

    /*!
     * \brief Starts the chronometer
     */
    void start()
    {
        end_ = std::chrono::steady_clock::now();
        start_ = std::chrono::steady_clock::now();
    }

    /*!
     * \brief Stops the chronometer
     */
    void stop() { end_ = std::chrono::steady_clock::now(); }

    /*!
     * \brief Get the seconds object
     *
     * \return size_t seconds elapsed
     */
    size_t get_seconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
    }

    /*!
     * \brief Get the milliseconds object
     *
     * \return size_t milliseconds elapsed
     */
    size_t get_milliseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }

    /*!
     * \brief Get the microseconds object
     *
     * \return size_t microseconds elapsed
     */
    size_t get_microseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
    }

    /*!
     * \brief Get the nanoseconds object
     *
     * \return size_t nanoseconds elapsed
     */
    size_t get_nanoseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
    }

  private:
    //! Current clock time when the Chrono object is started
    std::chrono::time_point<std::chrono::steady_clock> start_;
    //! Current clock time when the Chrono object is stopped
    std::chrono::time_point<std::chrono::steady_clock> end_;
};
