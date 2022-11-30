#include "API.hh"

namespace holovibes::api
{

void request_auto_contrast_all_windows()
{
    if (api::get_view_xy().contrast.auto_refresh)
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXY);

    if (api::get_cuts_view_enabled())
    {
        if (api::get_view_xz().contrast.auto_refresh)
            api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
        if (api::get_view_yz().contrast.auto_refresh)
            api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
    }

    if (api::get_filter2d_view_enabled())
        if (api::get_view_filter2d().contrast.auto_refresh)
            api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewFilter2D);
}

void request_exec_contrast_all_windows()
{
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXY);
    if (api::get_cuts_view_enabled())
    {
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
    }
    if (api::get_filter2d_view_enabled())
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewFilter2D);
}

void request_exec_contrast_current_window()
{
    if (api::is_current_view_xyz_type())
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(api::get_current_view_kind());
}

static float get_truncate_contrast_min(const int precision = 2)
{
    float value = api::get_current_view().get_contrast_min_logged();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_current_window_contrast_min(const float value)
{
    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_min();
    if (old_val != value)
        api::change_current_view()->contrast.min = get_current_view().log_enabled ? value : pow(10, value);
}

static float get_truncate_contrast_max(const int precision = 2)
{
    float value = api::get_current_view().get_contrast_max_logged();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_current_window_contrast_max(const float value)
{
    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_max();
    if (old_val != value)
        api::change_current_view()->contrast.max = get_current_view().log_enabled ? value : pow(10, value);
}

} // namespace holovibes::api
