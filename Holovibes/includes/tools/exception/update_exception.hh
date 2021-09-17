/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW_UPDATE_EXCEPTION(msg) throw holovibes::UpdateException("UpdateException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class UpdateException
 *
 * \brief #TODO Add a description for this class
 */
class UpdateException : public std::exception
{
  public:
    UpdateException(std::string msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~UpdateException() {}
};
} // namespace holovibes