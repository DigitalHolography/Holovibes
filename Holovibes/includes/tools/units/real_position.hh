/*! \file
 *
 * Implementation of a position in the frame desc coordinate system */
#pragma once

#include "unit.hh"

namespace holovibes
{
namespace units
{
class FDPixel;

/*! \brief A position in the frame desc coordinate system [0;fd.width]
 */
class RealPosition : public Unit<double>
{
  public:
    RealPosition(ConversionData data, Axis axis, double val = 0);
};
} // namespace units
} // namespace holovibes
