/*! \file
 *
 * \brief Implementation of a position in the Window coordinate system
 */
#pragma once

#include "unit.hh"

namespace holovibes::units
{
class FDPixel;
class OpenglPosition;

/*! \class WindowPixel
 *
 * \brief A position in the window coordinate system [0;window size]
 */
class WindowPixel : public Unit<int>
{
  public:
    WindowPixel(ConversionData data, Axis axis, int val = 0);

    operator OpenglPosition() const;
    operator FDPixel() const;
};
} // namespace holovibes::units
