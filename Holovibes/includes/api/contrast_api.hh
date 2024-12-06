/*! \file contrast_api.hh
 *
 * \brief Regroup all functions used to interact with the contrast and the log setting.
 *
 * Windows are XY, XZ, YZ and Filter2D.
 */
#pragma once

#include "API.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

#pragma region Internals

WindowKind get_current_window_type(); // Forward declaration

/*! \brief Returns the setting of the window kind specified
 *
 * \param[in] kind the window kind
 * \return ViewWindow the setting of the window kind
 */
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

/*! \brief Returns whether the log scale is enabled on the specified window kind
 *
 * \param[in] kind the window kind
 * \return bool true if the log scale is enabled
 */
inline bool get_log_enabled(WindowKind kind) { return get_window(kind).log_enabled; }

/*! \brief Returns whether the log scale is enabled on the current window
 *
 * \return bool true if the log scale is enabled
 */
inline bool get_log_enabled() { return get_log_enabled(get_current_window_type()); }

/*! \brief Enables or Disables log scale on the specified window kind
 *
 * \param[in] kind the window kind
 * \param[in] value true: enable, false: disable
 */
void set_log_enabled(WindowKind kind, bool value);

/*! \brief Enables or Disables log scale on the current window
 *
 * \param[in] value true: enable, false: disable
 */
inline void set_log_enabled(bool value) { return set_log_enabled(get_current_window_type(), value); }

#pragma endregion

#pragma region Contrast Range

/*! \brief Returns the contrast min value of the specified window kind.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] kind the window kind
 * \return float the contrast min value
 */
float get_contrast_min(WindowKind kind);

/*! \brief Returns the contrast max value of the focused window.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \return float the contrast max value
 */
inline float get_contrast_min() { return get_contrast_min(get_current_window_type()); }

/*! \brief Sets the contrast min value of the specified window kind.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] kind the window kind
 * \param[in] value the new contrast min value
 */
void set_contrast_min(WindowKind kind, float value);

/*! \brief Sets the contrast min value of the focused window.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] value the new contrast min value
 */
inline void set_contrast_min(float value) { return set_contrast_min(get_current_window_type(), value); }

/*! \brief Returns the contrast max value of the specified window kind.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] kind the window kind
 * \return float the contrast max value
 */
float get_contrast_max(WindowKind kind);

/*! \brief Returns the contrast max value of the focused window.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \return float the contrast max value
 */
inline float get_contrast_max() { return get_contrast_max(get_current_window_type()); }

/*! \brief Sets the contrast max value of the specified window kind.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] kind the window kind
 * \param[in] value the new contrast max value
 */
void set_contrast_max(WindowKind kind, float value);

/*! \brief Sets the contrast max value of the focused window.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] value the new contrast max value
 */
inline void set_contrast_max(float value) { return set_contrast_max(get_current_window_type(), value); }

/*! \brief Updates the contrast of the specified window kind.
 *
 * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
 *
 * \param[in] kind the window kind
 * \param[in] min the new contrast min value
 * \param[in] max the new contrast max value
 */
void update_contrast(WindowKind kind, float min, float max);

#pragma endregion

#pragma region Contrast Enabled

/*! \brief Returns whether the contrast is enabled on the specified window kind
 *
 * \param[in] kind the window kind
 * \return bool true if the contrast is enabled
 */
inline bool get_contrast_enabled(WindowKind kind) { return get_window(kind).contrast.enabled; }

/*! \brief Returns whether the contrast is enabled on the current window
 *
 * \return bool true if the contrast is enabled
 */
inline bool get_contrast_enabled() { return get_contrast_enabled(get_current_window_type()); }

/*! \brief Enables or Disables the contrast on the specified window kind
 *
 * \param[in] kind the window kind
 * \param[in] value true: enable, false: disable
 */
void set_contrast_enabled(WindowKind kind, bool value);

/*! \brief Enables or Disables the contrast on the current window
 *
 * \param[in] value true: enable, false: disable
 */
inline void set_contrast_enabled(bool value) { return set_contrast_enabled(get_current_window_type(), value); }

#pragma endregion

#pragma region Contrast Auto Refresh

/*! \brief Returns whether the contrast auto refresh is enabled on the specified window kind
 *
 * \param[in] kind the window kind
 * \return bool true if the contrast auto refresh is enabled
 */
inline bool get_contrast_auto_refresh(WindowKind kind) { return get_window(kind).contrast.auto_refresh; }

/*! \brief Returns whether the contrast auto refresh is enabled on the current window
 *
 * \return bool true if the contrast auto refresh is enabled
 */
inline bool get_contrast_auto_refresh() { return get_contrast_auto_refresh(get_current_window_type()); }

/*! \brief Enables or Disables the contrast auto refresh on the specified window kind
 *
 * \param[in] kind the window kind
 * \param[in] value true: enable, false: disable
 */
void set_contrast_auto_refresh(WindowKind kind, bool value);

/*! \brief Enables or Disables the contrast auto refresh on the current window
 *
 * \param[in] value true: enable, false: disable
 */
inline void set_contrast_auto_refresh(bool value)
{
    return set_contrast_auto_refresh(get_current_window_type(), value);
}

#pragma endregion

#pragma region Contrast Invert

/*! \brief Returns whether the contrast is inverted (min and max will be swaped) on the specified window kind
 *
 * \param[in] kind the window kind
 * \return bool true if the contrast is inverted
 */
inline bool get_contrast_invert(WindowKind kind) { return get_window(kind).contrast.invert; }

/*! \brief Returns whether the contrast is inverted (min and max will be swaped) on the current window
 *
 * \return bool true if the contrast is inverted
 */
inline bool get_contrast_invert() { return get_contrast_invert(get_current_window_type()); }

/*! \brief Enables or Disables the contrast invert on the specified window kind
 *
 * \param[in] kind the window kind
 * \param[in] value true: enable, false: disable
 */
void set_contrast_invert(WindowKind kind, bool value);

/*! \brief Enables or Disables the contrast invert on the current window
 *
 * \param[in] value true: enable, false: disable
 */
inline void set_contrast_invert(bool value) { return set_contrast_invert(get_current_window_type(), value); }

#pragma endregion

#pragma region Contrast Adv.

/*! \brief Returns the contrast lower threshold pourcentage. This setting is used to determine which percentile will be
 * used for the min contrast value when computing the auto contrast. Is in range [0, 100].
 *
 * \return float the contrast lower threshold
 */
inline float get_contrast_lower_threshold() { return GET_SETTING(ContrastLowerThreshold); }

/*! \brief Sets the contrast lower threshold pourcentage. This setting is used to determine which percentile will be
 * used for the min contrast value when computing the auto contrast. Must be in range [0, 100].
 *
 * \param[in] value the new contrast lower threshold in range [0, 100]
 */
inline void set_contrast_lower_threshold(float value) { UPDATE_SETTING(ContrastLowerThreshold, value); }

/*! \brief Returns the contrast upper threshold pourcentage. This setting is used to determine which percentile will be
 * used for the max contrast value when computing the auto contrast. Is in range [0, 100].
 *
 * \return float the contrast upper threshold
 */
inline float get_contrast_upper_threshold() { return GET_SETTING(ContrastUpperThreshold); }

/*! \brief Sets the contrast upper threshold pourcentage. This setting is used to determine which percentile will be
 * used for the max contrast value when computing the auto contrast. Must be in range [0, 100].
 *
 * \param[in] value the new contrast upper threshold in range [0, 100]
 */
inline void set_contrast_upper_threshold(float value) { UPDATE_SETTING(ContrastUpperThreshold, value); }

/*! \brief Returns the offset used to compute the autocontrast for the XZ and YZ views (cuts). Is in range [0,
 * get_fd().width / 2].
 *
 * The autocontrast will be calculated on a sub rect of the image where the sub rect is padded by offset in all
 * directions :
 * /---------------\
 * |               |
 * |    /-----\    |
 * |    |.....|    |
 * |    |.....|    |
 * |    \-----/    |
 * |               |
 * \---------------/
 *
 * \return uint the cuts contrast offset in range [0, get_fd().width / 2]
 */
inline uint get_cuts_contrast_p_offset() { return static_cast<uint>(GET_SETTING(CutsContrastPOffset)); }

/*! \brief Sets the offset used to compute the autocontrast for the XZ and YZ views (cuts). Must be in range [0,
 * get_fd().width / 2].
 *
 * The autocontrast will be calculated on a sub rect of the image where the sub rect is padded by offset in all
 * directions :
 * /---------------\
 * |               |
 * |    /-----\    |
 * |    |.....|    |
 * |    |.....|    |
 * |    \-----/    |
 * |               |
 * \---------------/
 *
 * \param[in] value the new cuts contrast offset in range [0, get_fd().width / 2]
 */
inline void set_cuts_contrast_p_offset(uint value) { UPDATE_SETTING(CutsContrastPOffset, value); }

#pragma endregion

#pragma region Reticle

/*! \brief Returns whether the reticle display is enabled. The reticle display is a rect region where the contrast is
 * calculated on.
 *
 * \return bool true if the reticle display is enabled
 */
inline bool get_reticle_display_enabled() { return GET_SETTING(ReticleDisplayEnabled); }

/*! \brief Enables or Disables the reticle display. The reticle display is a rect region where the contrast is
 * calculated on.
 *
 * \param[in] value true: enable, false: disable
 */
void set_reticle_display_enabled(bool value);

/*! \brief Returns the reticle scale. The reticle scale is the size of the reticle display.
 *
 * \return float the reticle scale
 */
inline float get_reticle_scale() { return GET_SETTING(ReticleScale); }

/*! \brief Sets the reticle scale. The reticle scale is the size of the reticle display.
 *
 * \param[in] value the new reticle scale
 */
void set_reticle_scale(float value);

/*! \brief Returns the reticle zone. This zone defines the rect region where the contrast is calculated on.
 *
 * \return units::RectFd the reticle zone
 */
inline units::RectFd get_reticle_zone() { return GET_SETTING(ReticleZone); };

/*! \brief Sets the reticle zone. This zone defines the rect region where the contrast is calculated on.
 *
 * \param[in] rect the new reticle zone
 */
inline void set_reticle_zone(const units::RectFd& rect) { UPDATE_SETTING(ReticleZone, rect); };

#pragma endregion

} // namespace holovibes::api