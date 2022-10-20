#include "API.hh"

namespace holovibes::api
{

void set_auto_contrast_all()
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    get_compute_pipe().request_autocontrast(WindowKind::XYview);
    if (api::get_cuts_view_enabled())
    {
        get_compute_pipe().request_autocontrast(WindowKind::XZview);
        get_compute_pipe().request_autocontrast(WindowKind::YZview);
    }
    if (api::get_filter2d_view_enabled())
        get_compute_pipe().request_autocontrast(WindowKind::Filter2D);

    pipe_refresh();
}

bool set_auto_contrast()
{
    try
    {
        get_compute_pipe().request_autocontrast(GSH::instance().get_value<CurrentWindowKind>());
        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
    }

    return false;
}

static float get_truncate_contrast_min(const int precision = 2)
{
    float value = api::get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_contrast_min(const double value)
{
    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_min();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        get_current_window().set_contrast_min(get_current_window().log_scale_slice_enabled ? value : pow(10, value));
        pipe_refresh();
    }
}

static float get_truncate_contrast_max(const int precision = 2)
{
    float value = api::get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_contrast_max(const double value)
{
    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_max();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        get_current_window().set_contrast_max(get_current_window().log_scale_slice_enabled ? value : pow(10, value));
        pipe_refresh();
    }
}

} // namespace holovibes::api
