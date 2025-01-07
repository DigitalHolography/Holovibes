#include "zoom_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "RawWindow.hh"
#include "HoloWindow.hh"

namespace holovibes::gui
{
ZoomOverlay::ZoomOverlay(BasicOpenGLWindow* parent)
    : SquareOverlay(KindOfOverlay::Zoom, parent)
{
    color_ = {0.f, .5f, 0.f};
}

void ZoomOverlay::release(ushort frameSide)
{
    display_ = false;

    if (zone_.top_left() == zone_.bottom_right())
        return;

    // handle Zoom
    RawWindow* window = dynamic_cast<RawWindow*>(parent_);
    if (window)
        window->zoomInRect(zone_);
}
} // namespace holovibes::gui
