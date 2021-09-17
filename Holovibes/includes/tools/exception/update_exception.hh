/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_UPDATE_EXCEPTION(msg) throw holovibes::UpdateException("UpdateException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class UpdateException
 *
 * \brief #TODO Add a description for this class
 */
class UpdateException : public CustomException
{
  public:
    UpdateException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    UpdateException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    ~UpdateException() {}
};
} // namespace holovibes