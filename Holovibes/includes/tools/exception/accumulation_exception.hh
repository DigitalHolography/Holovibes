/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once

#include "custom_exception.hh"

namespace holovibes
{
/*! \class AccumulationException
 *
 * \brief #TODO Add a description for this class
 */
class AccumulationException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes