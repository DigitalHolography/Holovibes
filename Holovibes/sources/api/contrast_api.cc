#include "contrast_api.hh"

namespace holovibes::api
{

#pragma region Internals

float ftruncate(const int precision, float value)
{
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

#pragma endregion

#pragma region Contrast

float get_contrast_min(WindowKind kind)
{
    float min = get_window(kind).contrast.min;
    return get_log_enabled(kind) ? min : log10(min);
}

float get_contrast_max(WindowKind kind)
{
    float max = get_window(kind).contrast.max;
    return get_log_enabled(kind) ? max : log10(max);
}

void set_contrast_min(WindowKind kind, float value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    // Get the minimum contrast value rounded for the comparison
    const float old_val = ftruncate(2, get_contrast_min(kind));
    if (old_val == value)
        return;

    float new_val = get_log_enabled(kind) ? value : pow(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.min, new_val);
        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.min = new_val;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void set_contrast_max(WindowKind kind, float value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    // Get the maximum contrast value rounded for the comparison
    const float old_val = ftruncate(2, get_contrast_max(kind));
    if (old_val == value)
        return;

    float new_val = get_log_enabled(kind) ? value : pow(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.max, new_val);
        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.max = new_val;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void update_contrast(WindowKind kind, float min, float max)
{
    min = min > 1.0f ? min : 1.0f;
    max = max > 1.0f ? max : 1.0f;

    if (kind == WindowKind::Filter2D)
    {
        auto window = GET_SETTING(Filter2d);
        window.contrast.min = min;
        window.contrast.max = max;
        UPDATE_SETTING(Filter2d, window);

        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.min = min;
    window.contrast.max = max;
    set_window_xyz(kind, window);
}

void set_contrast_enabled(WindowKind kind, bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.enabled, value);
        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.enabled = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void set_contrast_auto_refresh(WindowKind kind, bool value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.auto_refresh, value);
        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.auto_refresh = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void set_contrast_invert(WindowKind kind, bool value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.invert, value);
        return;
    }

    auto window = get_window_xyz(kind);
    window.contrast.invert = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

#pragma endregion

#pragma region Log

void set_log_enabled(WindowKind kind, const bool value)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, log_enabled, value);
        return;
    }

    auto window = get_window_xyz(kind);
    window.log_enabled = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

#pragma endregion

#pragma region Reticle

void set_reticle_display_enabled(bool value)
{
    if (get_reticle_display_enabled() == value)
        return;

    UPDATE_SETTING(ReticleDisplayEnabled, value);

    pipe_refresh();
}

void set_reticle_scale(float value)
{
    if (!is_between(value, 0.f, 1.f))
        return;

    UPDATE_SETTING(ReticleScale, value);

    pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api