#include "GUI.hh"
#include "user_interface_descriptor.hh"

#define UI UserInterfaceDescriptor::instance()

namespace holovibes::gui
{

void set_composite_area() { UI.mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>(); }

void open_advanced_settings(QMainWindow* parent)
{
    UI.is_advanced_settings_displayed = true;
    UI.advanced_settings_window_ = std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(parent);
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display() { return UI.mainDisplay; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz() { return UI.sliceXZ; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz() { return UI.sliceYZ; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window() { return UI.lens_window; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window() { return UI.raw_window; }

std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window() { return UI.filter2d_window; }

} // namespace holovibes::gui