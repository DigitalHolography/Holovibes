#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline WindowKind get_current_window_kind() { return api::detail::get_value<CurrentWindowKind>(); }
inline void change_current_window_kind(WindowKind index) { api::detail::change_value<CurrentWindowKind>(); }

inline bool is_current_window_xyz_type()
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(api::get_current_window_kind());
}

const View_Window& get_window(WindowKind kind);
inline const View_Window& get_current_window() { return get_window(get_current_window_kind()); }
inline const View_XYZ& get_current_window_as_view_xyz()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    return reinterpret_cast<const View_XYZ&>(api::get_current_window());
}

TriggerChangeValue<View_Window> change_window(WindowKind kind);
inline TriggerChangeValue<View_Window> change_current_window() { return change_window(get_current_window_kind()); }
inline TriggerChangeValue<View_XYZ> change_current_window_as_view_xyz()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    return TriggerChangeValue<View_XYZ>(api::change_current_window());
}

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
