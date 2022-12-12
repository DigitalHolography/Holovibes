#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline WindowKind get_current_view_kind() { return api::detail::get_value<CurrentWindowKind>(); }
inline void set_current_view_kind(WindowKind value) { api::detail::set_value<CurrentWindowKind>(value); }

inline bool is_view_xyz_type(WindowKind value)
{
    static const std::set<WindowKind> types = {WindowKind::ViewXY, WindowKind::ViewXZ, WindowKind::ViewYZ};
    return types.contains(value);
}
inline bool is_current_view_xyz_type() { return is_view_xyz_type(api::get_current_view_kind()); }

const ViewWindow& get_view(WindowKind kind);
inline const ViewXYZ& get_view_as_xyz_type(WindowKind kind)
{
    if (!is_view_xyz_type(kind))
        throw std::runtime_error("Bad window type when casting to a ViewXYZ");
    return reinterpret_cast<const ViewXYZ&>(api::get_view(kind));
}

inline const ViewWindow& get_current_view() { return get_view(api::get_current_view_kind()); }
inline const ViewXYZ& get_current_view_as_view_xyz() { return get_view_as_xyz_type(api::get_current_view_kind()); }

TriggerChangeValue<ViewWindow> change_view(WindowKind kind);
inline TriggerChangeValue<ViewXYZ> change_view_as_view_xyz(WindowKind kind)
{
    if (!is_view_xyz_type(kind))
        throw std::runtime_error("Bad window type when casting to a ViewXYZ");
    return TriggerChangeValue<ViewXYZ>(api::change_view(kind));
}
inline TriggerChangeValue<ViewWindow> change_current_view() { return change_view(api::get_current_view_kind()); }
inline TriggerChangeValue<ViewXYZ> change_current_view_as_view_xyz()
{
    return change_view_as_view_xyz(api::get_current_view_kind());
}

float get_z_boundary();

} // namespace holovibes::api
