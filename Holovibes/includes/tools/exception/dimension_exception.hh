#pragma once

#include "custom_exception.hh"

namespace holovibes
{

/*! \class CustomException
 *
 * \brief Upper exception class from which every holovibes exceptions should derive
 */
class DimensionException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes