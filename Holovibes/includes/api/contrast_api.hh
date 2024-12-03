/*! \file contrast_api.hh
 *
 * \brief Regroup all functions used to interact with the contrast and the log setting.
 *
 * Windows are XY, XZ, YZ and Filter2D.
 */
#pragma once

#include "common_api.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

#pragma region Internals

WindowKind get_current_window_type();

inline ViewWindow get_window(WindowKind kind)
{
    switch (kind)
    {
    case WindowKind::XYview:
        return GET_SETTING(XY);
    case WindowKind::XZview:
        return GET_SETTING(XZ);
    case WindowKind::YZview:
        return GET_SETTING(YZ);
    default:
        return GET_SETTING(Filter2d);
    }
}

#pragma endregion

#pragma region Log

inline bool get_log_enabled(WindowKind kind) { return get_window(kind).log_enabled; }
inline bool get_log_enabled() { return get_log_enabled(get_current_window_type()); }

void set_log_enabled(WindowKind kind, bool value);

/*! \brief Enables or Disables log scale on the current window
 *
 * \param value true: enable, false: disable
 */
inline void set_log_enabled(bool value) { return set_log_enabled(get_current_window_type(), value); }

#pragma endregion

#pragma region Contrast

inline bool get_contrast_enabled(WindowKind kind) { return get_window(kind).contrast.enabled; }
inline bool get_contrast_auto_refresh(WindowKind kind) { return get_window(kind).contrast.auto_refresh; }
inline bool get_contrast_invert(WindowKind kind) { return get_window(kind).contrast.invert; }
float get_contrast_min(WindowKind kind);
float get_contrast_max(WindowKind kind);

inline bool get_contrast_enabled() { return get_contrast_enabled(get_current_window_type()); }
inline bool get_contrast_auto_refresh() { return get_contrast_auto_refresh(get_current_window_type()); }
inline bool get_contrast_invert() { return get_contrast_invert(get_current_window_type()); }
inline float get_contrast_min() { return get_contrast_min(get_current_window_type()); }
inline float get_contrast_max() { return get_contrast_max(get_current_window_type()); }

void set_contrast_enabled(WindowKind kind, bool value);
void set_contrast_auto_refresh(WindowKind kind, bool value);
void set_contrast_invert(WindowKind kind, bool value);

void set_contrast_min(WindowKind kind, float value);
void set_contrast_max(WindowKind kind, float value);

void update_contrast(WindowKind kind, float min, float max);

inline void set_contrast_enabled(bool value) { return set_contrast_enabled(get_current_window_type(), value); }
inline void set_contrast_auto_refresh(bool value)
{
    return set_contrast_auto_refresh(get_current_window_type(), value);
}
inline void set_contrast_invert(bool value) { return set_contrast_invert(get_current_window_type(), value); }
inline void set_contrast_min(float value) { return set_contrast_min(get_current_window_type(), value); }
inline void set_contrast_max(float value) { return set_contrast_max(get_current_window_type(), value); }

#pragma endregion

#pragma region Contrast Adv.

inline float get_contrast_lower_threshold() { return GET_SETTING(ContrastLowerThreshold); }
inline void set_contrast_lower_threshold(float value) { UPDATE_SETTING(ContrastLowerThreshold, value); }

inline float get_contrast_upper_threshold() { return GET_SETTING(ContrastUpperThreshold); }
inline void set_contrast_upper_threshold(float value) { UPDATE_SETTING(ContrastUpperThreshold, value); }

inline uint get_cuts_contrast_p_offset() { return static_cast<uint>(GET_SETTING(CutsContrastPOffset)); }
inline void set_cuts_contrast_p_offset(uint value) { UPDATE_SETTING(CutsContrastPOffset, value); }

#pragma endregion

#pragma region Reticle

inline bool get_reticle_display_enabled() { return GET_SETTING(ReticleDisplayEnabled); }
void set_reticle_display_enabled(bool value);

inline float get_reticle_scale() { return GET_SETTING(ReticleScale); }
void set_reticle_scale(float value);

inline units::RectFd get_reticle_zone() { return GET_SETTING(ReticleZone); };
inline void set_reticle_zone(const units::RectFd& rect) { UPDATE_SETTING(ReticleZone, rect); };

#pragma endregion

} // namespace holovibes::api