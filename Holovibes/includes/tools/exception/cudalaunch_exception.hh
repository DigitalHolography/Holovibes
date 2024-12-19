/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once
#include <exception>
#include <string>
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_CUDALAUNCH_EXCEPTION(msg)                                                                                \
    throw holovibes::CudaLaunchException("CudaLaunchException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class CudaLaunchException
 *
 * \brief Custom exception class for cuda launch errors.
 */
class CudaLaunchException : public CustomException
{
  public:
    /*!
     * \brief Construct a new Cuda Launch Exception object
     *
     * \param msg the message you want to display with the exception
     */
    CudaLaunchException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    /*!
     * \brief Construct a new Cuda Launch Exception object
     *
     * \param msg the message you want to display with the exception
     * \param line the line from which the exception is triggered (__LINE__ in macro ONLY)
     * \param file the file from which the exception is triggered (__FILE__ in macro ONLY)
     */
    CudaLaunchException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    /*! \brief Destroy the Cuda Launch Exception object */
    ~CudaLaunchException() {}
};
} // namespace holovibes
