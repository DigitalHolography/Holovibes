/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once

#include "custom_exception.hh"

namespace holovibes
{
/*! \class UpdateException
 *
 * \brief User interactions update failure
 */
class UpdateException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes