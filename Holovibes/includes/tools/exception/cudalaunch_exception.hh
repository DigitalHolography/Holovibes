/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW_CUDALAUNCH_EXCEPTION(msg)                                                                                \
    throw holovibes::CudaLaunchException("CudaLaunchException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class CudaLaunchException
 *
 * \brief #TODO Add a description for this class
 */
class CudaLaunchException : public std::exception
{
  public:
    CudaLaunchException(std::string msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~CudaLaunchException() {}
};
} // namespace holovibes