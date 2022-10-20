#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

View_Window& get_current_window() { return GSH::instance().get_current_window(); }
std::shared_ptr<holovibes::View_Window> get_current_window_ptr() { return GSH::instance().get_current_window_ptr(); }

void change_window(const int index) { GSH::instance().change_window(index); }
std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
void start_chart_display();
void stop_chart_display();
double get_rotation();
bool get_flip_enabled();
unsigned int get_img_accu_level();
void set_raw_view(bool checked, uint auxiliary_window_max_size);
void set_lens_view(bool checked, uint auxiliary_window_max_size);
void close_windows();
void start_information_display(const std::function<void()>& callback)
{
    Holovibes::instance().start_information_display(callback);
}

bool is_current_window_xyz_type()
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(api::get_current_window_kind());
}

void set_composite_area()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>();
}

} // namespace holovibes::api
