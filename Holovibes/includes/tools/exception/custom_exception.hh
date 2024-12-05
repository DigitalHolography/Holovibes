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
 * \brief Upper exception class from which every holovibes exceptions should derive
 */
class CustomException : public std::exception
{
  public:
    /*!
     * \brief Construct a new Custom Exception object
     *
     * \param msg the message you want to display with the exception
     */
    CustomException(const std::string& msg)
        : std::exception(msg.c_str())
    {
        LOG_ERROR("Create except : {}", msg);
    }

    /*!
     * \brief Construct a new Custom Exception object
     *
     * \param msg the message you want to display with the exception
     * \param line the line from which the exception is triggered (__LINE__ in macro ONLY)
     * \param file the file from which the exception is triggered (__FILE__ in macro ONLY)
     */
    CustomException(const std::string& msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR("{} {}:{}", msg, file, line);
    }

    /*! \brief Destroy the Custom Exception object */
    ~CustomException() {}
};
} // namespace holovibes
