#include "gui_utilities.hh"

namespace holovibes::gui::utilities
{
void display_reticle(bool value)
{
    api::change_reticle()->display_enabled = value;

    if (value)
    {
        UserInterface::instance().main_display->getOverlayManager().create_overlay<gui::KindOfOverlay::Reticle>();
        UserInterface::instance().main_display->getOverlayManager().create_default();
    }
    else
        UserInterface::instance().main_display->getOverlayManager().disable_all(gui::KindOfOverlay::Reticle);
}
} // namespace holovibes::gui::utilities