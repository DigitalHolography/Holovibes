/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_REFRESH_EXCEPTION(msg) throw holovibes::RefreshException("RefreshException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class RefreshException
 *
 * \brief #TODO Add a description for this class
 */
class RefreshException : public CustomException
{
  public:
    RefreshException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    RefreshException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    ~RefreshException() {}
};
} // namespace holovibes