/*! \file
 *
 * Implementation of a position in the frame desc coordinate system */
#pragma once

#include "unit.hh"

namespace holovibes
{
namespace units
{
class WindowPixel;
class OpenglPosition;
class RealPosition;

/*! \brief A position in the frame desc coordinate system [0;fd.width]
 */
class FDPixel : public Unit<int>
{
  public:
    FDPixel(ConversionData data, Axis axis, int val = 0);

    operator OpenglPosition() const;
    operator WindowPixel() const;
    operator RealPosition() const;
};
} // namespace units
} // namespace holovibes
