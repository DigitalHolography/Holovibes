/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Implementation of a position in the Window coordinate system */
#pragma once

#include "unit.hh"

namespace holovibes
{
namespace units
{
class FDPixel;
class OpenglPosition;

/*! \brief A position in the window coordinate system [0;window size]
 */
class WindowPixel : public Unit<int>
{
  public:
    WindowPixel(ConversionData data, Axis axis, int val = 0);

    operator OpenglPosition() const;
    operator FDPixel() const;
};
} // namespace units
} // namespace holovibes
