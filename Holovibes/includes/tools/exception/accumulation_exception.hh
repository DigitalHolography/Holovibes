/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW_ACCUMULATION_EXCEPTION(msg)                                                                              \
    throw holovibes::AccumulationException("AccumulationException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class AccumulationException
 *
 * \brief #TODO Add a description for this class
 */
class AccumulationException : public std::exception
{
  public:
    AccumulationException(std::string msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~AccumulationException() {}
};
} // namespace holovibes