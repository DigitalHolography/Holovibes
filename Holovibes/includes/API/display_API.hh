#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline WindowKind get_current_window_kind() { return api::detail::get_value<CurrentWindowKind>(); }
inline void set_current_window_kind(WindowKind value) { api::detail::set_value<CurrentWindowKind>(value); }

inline bool is_current_window_xyz_type()
{
    static const std::set<WindowKind> types = {WindowKind::ViewXY, WindowKind::ViewXZ, WindowKind::ViewYZ};
    return types.contains(api::get_current_window_kind());
}

const ViewWindow& get_window(WindowKind kind);
inline const ViewWindow& get_current_window() { return get_window(get_current_window_kind()); }
inline const ViewXYZ& get_current_window_as_view_xyz()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    return reinterpret_cast<const ViewXYZ&>(api::get_current_window());
}

TriggerChangeValue<ViewWindow> change_window(WindowKind kind);
inline TriggerChangeValue<ViewWindow> change_current_window() { return change_window(get_current_window_kind()); }
inline TriggerChangeValue<ViewXYZ> change_current_window_as_view_xyz()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    return TriggerChangeValue<ViewXYZ>(api::change_current_window());
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

// FIXME : TO DELETE
inline void set_composite_area()
{
    UserInterfaceDescriptor::instance()
        .mainDisplay->getOverlayManager()
        .create_overlay<gui::KindOfOverlay::CompositeArea>();
}

} // namespace holovibes::api
