#include "contrast_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Internals

float ftruncate(const int precision, float value)
{
    const float multiplier = std::powf(10.0f, static_cast<float>(precision));
    return std::round(value * multiplier) / multiplier;
}

ContrastRange ContrastApi::get_contrast_range(WindowKind kind) const
{
    switch (kind)
    {
    case WindowKind::XYview:
        return GET_SETTING(XYContrastRange);
    case WindowKind::XZview:
        return GET_SETTING(XZContrastRange);
    case WindowKind::YZview:
        return GET_SETTING(YZContrastRange);
    default:
        return GET_SETTING(Filter2dContrastRange);
    }
}

void ContrastApi::set_contrast_range(ContrastRange range, WindowKind kind) const
{
    switch (kind)
    {
    case WindowKind::XYview:
        UPDATE_SETTING(XYContrastRange, range);
        break;
    case WindowKind::XZview:
        UPDATE_SETTING(XZContrastRange, range);
        break;
    case WindowKind::YZview:
        UPDATE_SETTING(YZContrastRange, range);
        break;
    default:
        UPDATE_SETTING(Filter2dContrastRange, range);
        break;
    }
}

#pragma endregion

#pragma region Contrast

float ContrastApi::get_contrast_min(WindowKind kind) const
{
    float min = get_contrast_range(kind).min;
    return get_log_enabled(kind) ? min : log10(min);
}

void ContrastApi::set_contrast_min(float value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || !get_contrast_enabled())
        return;

    // Get the minimum contrast value rounded for the comparison
    const float old_val = ftruncate(2, get_contrast_min(kind));
    if (old_val == value)
        return;

    float new_val = get_log_enabled(kind) ? value : powf(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.min, new_val);
        return;
    }

    auto contrast_range = get_contrast_range(kind);
    contrast_range.min = new_val;
    set_contrast_range(contrast_range, kind);

    api_->compute.pipe_refresh();
}

float ContrastApi::get_contrast_max(WindowKind kind) const
{
    float max = get_contrast_range(kind).max;
    return get_log_enabled(kind) ? max : log10(max);
}

void ContrastApi::set_contrast_max(float value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || !get_contrast_enabled())
        return;

    // Get the maximum contrast value rounded for the comparison
    const float old_val = ftruncate(2, get_contrast_max(kind));
    if (old_val == value)
        return;

    float new_val = get_log_enabled(kind) ? value : powf(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.max, new_val);
        return;
    }

    auto contrast_range = get_contrast_range(kind);
    contrast_range.max = new_val;
    set_contrast_range(contrast_range, kind);

    api_->compute.pipe_refresh();
}

void ContrastApi::update_contrast(float min, float max, WindowKind kind) const
{
    min = min > 1.0f ? min : 1.0f;
    max = max > 1.0f ? max : 1.0f;

    auto contrast_range = get_contrast_range(kind);
    contrast_range.min = min;
    contrast_range.max = max;
    set_contrast_range(contrast_range, kind);
}

#pragma endregion

#pragma region Contrast Enabled

void ContrastApi::set_contrast_enabled(bool value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.enabled, value);
        return;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.contrast.enabled = value;
    api_->window_pp.set_window_xyz(kind, window);

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Contrast Auto Refresh

void ContrastApi::set_contrast_auto_refresh(bool value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || !get_contrast_enabled())
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.auto_refresh, value);
        return;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.contrast.auto_refresh = value;
    api_->window_pp.set_window_xyz(kind, window);

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Contrast Invert

bool ContrastApi::get_contrast_invert(WindowKind kind) const { return get_contrast_range(kind).invert; }

void ContrastApi::set_contrast_invert(bool value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || !get_contrast_enabled())
        return;

    auto contrast_range = get_contrast_range(kind);
    contrast_range.invert = value;
    set_contrast_range(contrast_range, kind);

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Log

void ContrastApi::set_log_enabled(const bool value, WindowKind kind) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, log_enabled, value);
        return;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.log_enabled = value;
    api_->window_pp.set_window_xyz(kind, window);

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Reticle

void ContrastApi::set_reticle_display_enabled(bool value) const
{
    if (get_reticle_display_enabled() == value)
        return;

    UPDATE_SETTING(ReticleDisplayEnabled, value);

    // api_->compute.pipe_refresh();
}

void ContrastApi::set_reticle_scale(float value) const
{
    if (!is_between(value, 0.f, 1.f))
        return;

    UPDATE_SETTING(ReticleScale, value);

    api_->compute.pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api