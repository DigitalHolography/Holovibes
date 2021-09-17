/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_CUDALAUNCH_EXCEPTION(msg)                                                                                \
    throw holovibes::CudaLaunchException("CudaLaunchException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class CudaLaunchException
 *
 * \brief #TODO Add a description for this class
 */
class CudaLaunchException : public CustomException
{
  public:
    CudaLaunchException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    CudaLaunchException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    ~CudaLaunchException() {}
};
} // namespace holovibes