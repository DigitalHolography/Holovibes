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
    EnqueueException(const std::string& msg)
        : CustomException(msg.c_str())
    {
    }

    EnqueueException(const std::string& msg, const int line, const char* file)
        : CustomException(msg.c_str(), line, file)
    {
    }

    ~EnqueueException() {}
};
} // namespace holovibes