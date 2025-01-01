/*! \file
 *
 * \brief Implementation of a position in the frame desc coordinate system
 */
#pragma once

#include "unit.hh"

namespace holovibes::units
{
class OpenglPosition;

/*! \class FDPixel
 *
 * \brief A position in the frame desc coordinate system [0;fd.width]
 */
class FDPixel : public Unit<int>
{
  public:
    FDPixel(ConversionData data, Axis axis, int val = 0);

    operator OpenglPosition() const;
};
} // namespace holovibes::units
