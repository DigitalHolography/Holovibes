#include "API.hh"
#include "composite_area_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include <iostream>

using holovibes::gui::BasicOpenGLWindow;
using holovibes::gui::CompositeAreaOverlay;

CompositeAreaOverlay::CompositeAreaOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::CompositeArea, parent)
{
    color_ = {0.6f, 0.5f, 0.0f};
}

void CompositeAreaOverlay::release(ushort frameSide)
{
    disable();

    if (zone_.topLeft() == zone_.bottomRight())
        return;

    API.composite.set_composite_zone(zone_);
}
