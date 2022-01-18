/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once
#include "custom_exception.hh"

namespace holovibes
{
/*! \class EnqueueException
 *
 * \brief Exceptions for queues
 */
class EnqueueException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes