/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_ACCUMULATION_EXCEPTION(msg)                                                                              \
    throw holovibes::AccumulationException("AccumulationException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class AccumulationException
 *
 * \brief #TODO Add a description for this class
 */
class AccumulationException : public CustomException
{
  public:
    AccumulationException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    AccumulationException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    ~AccumulationException() {}
};
} // namespace holovibes