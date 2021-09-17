/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW(msg) throw holovibes::CustomException(msg, __LINE__, __FILE__)

namespace holovibes
{

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
    {
        LOG_ERROR << msg;
    }

    // NEW
    CustomException(const std::string& msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~CustomException() {}
};
} // namespace holovibes