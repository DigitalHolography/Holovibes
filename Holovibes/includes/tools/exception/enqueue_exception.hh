/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"
#include "custom_exception.hh"

#define THROW_ENQUEUE_EXCEPTION(msg) throw holovibes::EnqueueException("EnqueueException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class EnqueueException
 *
 * \brief #TODO Add a description for this class
 */
class EnqueueException : public CustomException
{
  public:
    /*!
     * \brief Construct a new Enqueue Exception object
     *
     * \param msg the message you want to display with the exception
     */
    EnqueueException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    /*!
     * \brief Construct a new Enqueue Exception object
     *
     * \param msg the message you want to display with the exception
     * \param line the line from which the exception is triggered (__LINE__ in macro ONLY)
     * \param file the file from which the exception is triggered (__FILE__ in macro ONLY)
     */
    EnqueueException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    /*! \brief Destroy the Enqueue Exception object */
    ~EnqueueException() {}
};
} // namespace holovibes
