/*! \file
 *
 * Implementation of custom error class. */
#include <exception>
#include <string>
#pragma once

namespace holovibes
{
/*! \brief Implementation of custom error class.
 *
 * To create a new kind of error just add your new kind of error to the enum */

enum error_kind
{
    fail_update,
    fail_refresh,
    fail_accumulation,
    fail_cudaLaunch,
    fail_enqueue
};

class CustomException : public std::exception
{
  public:
    CustomException(std::string msg, const error_kind& kind)
        : std::exception(msg.c_str())
        , error_kind_(kind)
    {
    }

    ~CustomException() {}

    const error_kind& get_kind() const { return error_kind_; }

  private:
    const error_kind& error_kind_;
};
} // namespace holovibes