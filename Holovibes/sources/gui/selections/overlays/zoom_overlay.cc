/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "zoom_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "RawWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
namespace gui
{
ZoomOverlay::ZoomOverlay(BasicOpenGLWindow* parent)
    : SquareOverlay(KindOfOverlay::Zoom, parent)
{
    color_ = {0.f, 0.5f, 0.f};
}

void ZoomOverlay::release(ushort frameSide)
{
    if (zone_.topLeft() == zone_.bottomRight())
    {
        disable();
        return;
    }

    // handle Zoom
    RawWindow* window = dynamic_cast<RawWindow*>(parent_);
    if (window)
    {
        window->zoomInRect(zone_);
    }
    disable();
}
} // namespace gui
} // namespace holovibes
