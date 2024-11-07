#pragma once

#include "overlay_manager.hh"

namespace holovibes::gui
{

template <KindOfOverlay ko>
void OverlayManager::enable(bool make_current, int ms)
{
    // Find the overlay in the vector
    std::shared_ptr<Overlay> new_overlay = nullptr;
    bool found = false;
    for (auto o : overlays_)
        if (o->getKind() == ko)
        {
            new_overlay = o;
            found = true;
            break;
        }

    // If the overlay is not found, create it
    if (!found)
    {
        new_overlay = create_overlay(ko);
        clean();
        overlays_.push_back(new_overlay);
        new_overlay->initProgram();
    }

    // Update the time before hide
    if (ms > 0)
        new_overlay->setTimeBeforeHide(QDateTime::currentDateTime().addMSecs(ms));
    else
        new_overlay->setTimeBeforeHide(QDateTime::currentDateTime().addYears(1));

    new_overlay->enable();

    if (make_current)
    {
        current_overlay_ = new_overlay;
        current_overlay_->onSetCurrent();
    }
}

} // namespace holovibes::gui
