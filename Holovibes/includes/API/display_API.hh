#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline View_Window& get_current_window() { return GSH::instance().get_current_window(); }

inline bool is_current_window_xyz_type()
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(api::get_current_window_kind());
}

inline View_XYZ& get_current_window_as_view_xyz()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return reinterpret_cast<View_XYZ&>(api::get_current_window());
}

inline void change_window(WindowKind index) { GSH::instance().change_window(index); }

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
void start_chart_display();
void stop_chart_display();
void set_raw_view(bool checked, uint auxiliary_window_max_size);
void set_lens_view(bool checked, uint auxiliary_window_max_size);
void close_windows();

inline void start_information_display(const std::function<void()>& callback)
{
    Holovibes::instance().start_information_display(callback);
}

inline void set_composite_area()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>();
}

} // namespace holovibes::api
