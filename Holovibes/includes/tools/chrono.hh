/*! \file
 *
 * \brief Wrapper around the chrono library
 */
#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>

/*! \class Chrono
 *
 * \brief Wrapper around the chrono library
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

    /*! \brief Stops the chronometer */
    void stop() { end_ = std::chrono::steady_clock::now(); }

    void wait(double seconds)
    {
        auto duration = std::chrono::duration<double>(seconds);

        while (std::chrono::steady_clock::now() - start_ < duration){}
        
        stop();
    }

    /*! \brief Get the seconds object
     *
     * \return size_t seconds elapsed
     */
    size_t get_seconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
    }

    /*! \brief Get the milliseconds object
     *
     * \return size_t milliseconds elapsed
     */
    size_t get_milliseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }

    /*! \brief Get the microseconds object
     *
     * \return size_t microseconds elapsed
     */
    size_t get_microseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
    }

    /*! \brief Get the nanoseconds object
     *
     * \return size_t nanoseconds elapsed
     */
    size_t get_nanoseconds()
    {
        if (end_ <= start_)
            stop();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
    }

    /*!
     * \brief Get the current, formatted date
     */
    static std::string get_current_date()
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        std::tm* timeinfo = std::localtime(&in_time_t);
        int year = timeinfo->tm_year % 100;
        ss << std::setw(2) << std::setfill('0') << year << std::put_time(timeinfo, "%m%d");
        return ss.str();
    }

    /*!
     * \brief Get the current, formatted date-time.
     */
    static std::string get_current_date_time()
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%Hh%M-%S");
        return ss.str();
    }

  private:
    /*! \brief Current clock time when the Chrono object is started */
    std::chrono::time_point<std::chrono::steady_clock> start_;
    /*! \brief Current clock time when the Chrono object is stopped */
    std::chrono::time_point<std::chrono::steady_clock> end_;
};
