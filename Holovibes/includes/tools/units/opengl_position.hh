/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Implementation of a position in the OpenGl coordinate system */
#pragma once

#include "unit.hh"

namespace holovibes
{
namespace units
{
class FDPixel;
class WindowPixel;

/*! \brief A position in the OpenGL coordinate system [-1;1]
 */
class OpenglPosition : public Unit<float>
{
  public:
    OpenglPosition(ConversionData data, Axis axis, float val = 0);

    operator FDPixel() const;
    operator WindowPixel() const;
};
} // namespace units
} // namespace holovibes