/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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

    parent_->getCd()->setCompositeZone(zone_);
}
