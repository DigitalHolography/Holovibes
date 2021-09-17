/*! \file
 *
 * \brief Implementation of custom error class.
 */
#include <exception>
#include <string>
#pragma once
#include "logger.hh"

#define THROW_ENQUEUE_EXCEPTION(msg) throw holovibes::EnqueueException("EnqueueException: "##msg, __LINE__, __FILE__)

namespace holovibes
{
/*! \class EnqueueException
 *
 * \brief #TODO Add a description for this class
 */
class EnqueueException : public std::exception
{
  public:
    EnqueueException(std::string msg, const int line, const char* file)
        : std::exception(msg.c_str())
    {
        LOG_ERROR << msg << " " << file << ':' << line;
    }

    ~EnqueueException() {}
};
} // namespace holovibes