/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW(msg, error_kind) throw holovibes::CustomException(msg, error_kind, __LINE__, __FILE__)

namespace holovibes
{
/*! \enum error_kind
 *
 * \brief Implementation of custom error class.
 *
 * To create a new kind of error just add your new kind of error to the enum
 */
enum error_kind
{
    fail_update,
    fail_refresh,
    fail_accumulation,
    fail_cudaLaunch,
    fail_enqueue
};

/*! \class CustomException
 *
 * \brief #TODO Add a description for this class
 */
class CustomException : public std::exception
{
  public:
    // NEW
    CustomException(const std::string& msg)
        : std::exception(msg.c_str())
        , error_kind_(error_kind::fail_update)
    {
        LOG_ERROR << msg;
    }

    // NEW
    CustomException(const std::string& msg, const int line, const char* file)
        : std::exception(msg.c_str())
        , error_kind_(error_kind::fail_update)
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    // OLD
    CustomException(std::string msg, const error_kind& kind)
        : std::exception(msg.c_str())
        , error_kind_(kind)
    {
        LOG_ERROR << "CustomException have been thrown";
    }

    // OLD
    CustomException(std::string msg, const error_kind& kind, const int line, const char* file)
        : std::exception(msg.c_str())
        , error_kind_(kind)
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~CustomException() {}

    const error_kind& get_kind() const { return error_kind_; }

  private:
    const error_kind& error_kind_;
};
} // namespace holovibes