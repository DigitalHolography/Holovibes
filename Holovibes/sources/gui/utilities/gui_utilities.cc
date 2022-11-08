#include "gui_utilities.hh"

namespace holovibes::gui::utilities
{
void display_reticle(bool value)
{
    api::change_reticle()->display_enabled = value;

    if (value)
    {
        UserInterfaceDescriptor::instance()
            .mainDisplay->getOverlayManager()
            .create_overlay<gui::KindOfOverlay::Reticle>();
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_default();
    }
    else
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::KindOfOverlay::Reticle);
}
} // namespace holovibes::gui