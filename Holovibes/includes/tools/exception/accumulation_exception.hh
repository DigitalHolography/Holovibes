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
 * \brief Custom exception class for accumulation errors.
 */
class AccumulationException : public CustomException
{
  public:
    /*!
     * \brief Construct a new Accumulation Exception object
     *
     * \param msg the message you want to display with the exception
     */
    AccumulationException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    /*!
     * \brief Construct a new Accumulation Exception object
     *
     * \param msg the message you want to display with the exception
     * \param line the line from which the exception is triggered (__LINE__ in macro ONLY)
     * \param file the file from which the exception is triggered (__FILE__ in macro ONLY)
     */
    AccumulationException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    /*! \brief Destroy the Accumulation Exception object */
    ~AccumulationException() {}
};
} // namespace holovibes
