/*! \file
 *
 * \brief Implementation of a position in the OpenGl coordinate system
 */
#pragma once

#include "unit.hh"

namespace holovibes::units
{
class FDPixel;
class WindowPixel;

/*! \class OpenglPosition
 *
 * \brief A position in the OpenGL coordinate system [-1;1]
 */
class OpenglPosition : public Unit<float>
{
  public:
    OpenglPosition(ConversionData data, Axis axis, float val = 0);

    operator FDPixel() const;
    operator WindowPixel() const;
};
} // namespace holovibes::units