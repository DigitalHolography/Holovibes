/*! \file
 *
 * \brief Implementation of custom error class.
 */
#pragma once
#include <exception>
#include <string>
#include "logger.hh"
#include "custom_exception.hh"

namespace holovibes
{
/*! \class CudaLaunchException
 *
 * \brief Exception during initialization of the graphic card
 */
class CudaLaunchException : public CustomException
{
  public:
    using CustomException::CustomException;
};
} // namespace holovibes