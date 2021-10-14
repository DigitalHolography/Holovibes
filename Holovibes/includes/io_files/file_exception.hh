/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <exception>
#include <string>

namespace holovibes::io_files
{
/*! \class FileException
 *
 * \brief #TODO Add a description for this class
 */
class FileException : public std::exception
{
  public:
    /*! \brief Default constructor
     *
     * \param error_msg The message error of the exception
     * \param display_errno Should the error message specify the errno
     */
    FileException(const std::string& error_msg, bool display_errno = true);

    const char* what() const noexcept override { return error_msg_.c_str(); }

  private:
    std::string error_msg_;
};
} // namespace holovibes::io_files
