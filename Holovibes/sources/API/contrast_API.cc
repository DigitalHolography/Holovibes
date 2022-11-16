#include "API.hh"

namespace holovibes::api
{

void request_auto_contrast_all_windows()
{
    // FIXME API : Need to move this outside this
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    // FIXME API : this code stay here
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXY);
    if (api::get_cuts_view_enabled())
    {
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
    }
    if (api::get_filter2d_view_enabled())
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewFilter2D);
}

void request_auto_contrast_current_window()
{
    if (api::is_current_window_xyz_type())
        api::get_compute_pipe().get_rendering().request_view_exec_contrast(api::get_current_window_kind());
}

static float get_truncate_contrast_min(const int precision = 2)
{
    float value = api::get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_current_window_contrast_min(const float value)
{
    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_min();
    if (old_val != value)
        api::change_current_window()->contrast.min = get_current_window().log_enabled ? value : pow(10, value);
}

static float get_truncate_contrast_max(const int precision = 2)
{
    float value = api::get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_current_window_contrast_max(const float value)
{
    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_max();
    if (old_val != value)
        api::change_current_window()->contrast.max = get_current_window().log_enabled ? value : pow(10, value);
}

} // namespace holovibes::api