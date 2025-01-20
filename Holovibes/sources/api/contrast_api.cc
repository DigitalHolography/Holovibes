#include "contrast_api.hh"

#include "API.hh"

#define NOT_SAME_AND_NOT_RAW(old_val, new_val)                                                                         \
    if (old_val == new_val)                                                                                            \
        return ApiCode::NO_CHANGE;                                                                                     \
    if (api_->compute.get_compute_mode() == Computation::Raw)                                                          \
        return ApiCode::WRONG_COMP_MODE;

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

#pragma region Log

ApiCode ContrastApi::set_log_enabled(const bool value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(get_log_enabled(kind), value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, log_enabled, value);
        return ApiCode::OK;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.log_enabled = value;
    api_->window_pp.set_window_xyz(kind, window);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Contrast

float ContrastApi::get_contrast_min(WindowKind kind) const
{
    float min = get_contrast_range(kind).min;
    return get_log_enabled(kind) ? min : log10(min);
}

ApiCode ContrastApi::set_contrast_min(float value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(ftruncate(2, get_contrast_min(kind)), value);

    if (!get_contrast_enabled())
        return ApiCode::INVALID_VALUE;

    float new_val = get_log_enabled(kind) ? value : powf(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.min, new_val);
        return ApiCode::OK;
    }

    auto contrast_range = get_contrast_range(kind);
    contrast_range.min = new_val;
    set_contrast_range(contrast_range, kind);

    return ApiCode::OK;
}

float ContrastApi::get_contrast_max(WindowKind kind) const
{
    float max = get_contrast_range(kind).max;
    return get_log_enabled(kind) ? max : log10(max);
}

ApiCode ContrastApi::set_contrast_max(float value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(ftruncate(2, get_contrast_max(kind)), value);

    if (!get_contrast_enabled())
        return ApiCode::INVALID_VALUE;

    float new_val = get_log_enabled(kind) ? value : powf(10, value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.max, new_val);
        return ApiCode::OK;
    }

    auto contrast_range = get_contrast_range(kind);
    contrast_range.max = new_val;
    set_contrast_range(contrast_range, kind);

    return ApiCode::OK;
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

ApiCode ContrastApi::set_contrast_enabled(bool value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(get_contrast_enabled(kind), value);

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.enabled, value);
        return ApiCode::OK;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.contrast.enabled = value;
    api_->window_pp.set_window_xyz(kind, window);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Contrast Auto Refresh

ApiCode ContrastApi::set_contrast_auto_refresh(bool value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(get_contrast_auto_refresh(kind), value);

    if (!get_contrast_enabled())
        return ApiCode::INVALID_VALUE;

    if (kind == WindowKind::Filter2D)
    {
        SET_SETTING(Filter2d, contrast.auto_refresh, value);
        return ApiCode::OK;
    }

    auto window = api_->window_pp.get_window_xyz(kind);
    window.contrast.auto_refresh = value;
    api_->window_pp.set_window_xyz(kind, window);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Contrast Invert

bool ContrastApi::get_contrast_invert(WindowKind kind) const { return get_contrast_range(kind).invert; }

ApiCode ContrastApi::set_contrast_invert(bool value, WindowKind kind) const
{
    NOT_SAME_AND_NOT_RAW(get_contrast_invert(kind), value);

    if (!get_contrast_enabled())
        return ApiCode::INVALID_VALUE;

    auto contrast_range = get_contrast_range(kind);
    contrast_range.invert = value;
    set_contrast_range(contrast_range, kind);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Contrast Adv.

ApiCode ContrastApi::set_contrast_lower_threshold(float value) const
{
    if (get_contrast_lower_threshold() == value)
        return ApiCode::NO_CHANGE;

    if (!is_between(value, 0.f, 100.f))
    {
        LOG_WARN("Contrast lower threshold must be in range [0., 100.]");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(ContrastLowerThreshold, value);

    return ApiCode::OK;
}

ApiCode ContrastApi::set_contrast_upper_threshold(float value) const
{
    if (get_contrast_upper_threshold() == value)
        return ApiCode::NO_CHANGE;

    if (!is_between(value, 0.f, 100.f))
    {
        LOG_WARN("Contrast upper threshold must be in range [0., 100.]");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(ContrastUpperThreshold, value);

    return ApiCode::OK;
}

ApiCode ContrastApi::set_cuts_contrast_p_offset(uint value) const
{
    if (get_contrast_upper_threshold() == value)
        return ApiCode::NO_CHANGE;

    if (api_->input.get_import_type() != ImportType::None &&
        !is_between(static_cast<float>(value), 0.f, api_->input.get_input_fd().width / 2.f))
    {
        LOG_WARN("Contrast upper threshold must be in range [0, get_fd().width / 2]");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(CutsContrastPOffset, value);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Reticle

ApiCode ContrastApi::set_reticle_display_enabled(bool value) const
{
    NOT_SAME_AND_NOT_RAW(get_reticle_display_enabled(), value);

    UPDATE_SETTING(ReticleDisplayEnabled, value);

    return ApiCode::OK;
}

ApiCode ContrastApi::set_reticle_scale(float value) const
{
    NOT_SAME_AND_NOT_RAW(get_reticle_scale(), value);

    if (!get_reticle_display_enabled())
    {
        LOG_WARN("Reticle display must be enabled to set the reticle scale");
        return ApiCode::INVALID_VALUE;
    }

    if (!is_between(value, 0.f, 1.f))
    {
        LOG_WARN("Reticle scale must be in range [0., 1.]");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(ReticleScale, value);

    return ApiCode::OK;
}

ApiCode ContrastApi::set_reticle_zone(const units::RectFd& rect) const
{
    NOT_SAME_AND_NOT_RAW(get_reticle_zone(), rect);

    if (!get_reticle_display_enabled())
    {
        LOG_WARN("Reticle display must be enabled to set the reticle zone");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(ReticleZone, rect);

    return ApiCode::OK;
};

#pragma endregion

} // namespace holovibes::api