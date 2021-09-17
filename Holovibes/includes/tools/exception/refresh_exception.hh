/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW_REFRESH_EXCEPTION(msg) throw holovibes::RefreshException("RefreshException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class RefreshException
 *
 * \brief #TODO Add a description for this class
 */
class RefreshException : public std::exception
{
  public:
    RefreshException(std::string msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~RefreshException() {}
};
} // namespace holovibes