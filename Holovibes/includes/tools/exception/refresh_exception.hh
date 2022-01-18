/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once
#include "custom_exception.hh"

namespace holovibes
{
/*! \class RefreshException
 *
 * \brief Exception during the refresh of the pipe
 */
class RefreshException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes