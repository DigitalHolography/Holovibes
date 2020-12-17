/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Overlay selecting the zone to normalize colors. */
#pragma once

#include "square_overlay.hh"

namespace holovibes
{
namespace gui
{
class CompositeAreaOverlay : public RectOverlay
{
  public:
    CompositeAreaOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace gui
} // namespace holovibes
