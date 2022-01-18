/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once

#include <exception>
#include <string>
#include "logger.hh"

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
    explicit CustomException(const std::string& msg)
        : msg_(msg)
    {
        LOG_ERROR(main, "Create except : {}", msg);
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
        LOG_ERROR(main, "{} {}:{}", msg, file, line);
    }

    const char* what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};
} // namespace holovibes
