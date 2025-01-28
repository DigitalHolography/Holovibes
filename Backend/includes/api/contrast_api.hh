/*! \file contrast_api.hh
 *
 * \brief Regroup all functions used to interact with the contrast and the log setting.
 *
 * Windows are XY, XZ, YZ and Filter2D.
 */
#pragma once

#include "enum_window_kind.hh"
#include "common_api.hh"

namespace holovibes::api
{

/*! \class ContrastApi
 *
 * \brief Regroup all functions related to contrast (min, max, auto, invert, low/high treshold), log and reticle
 * settings.
 */
class ContrastApi : public IApi
{

  public:
    ContrastApi(const Api* api)
        : IApi(api)
    {
    }

  private:
#pragma region Internals

    /*! \brief Returns the setting of the window kind specified
     *
     * \param[in] kind the window kind
     * \return ViewWindow the setting of the window kind
     */
    inline ViewWindow get_window(WindowKind kind) const
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

    /*! \brief Returns the type of the focused window. This setting is only useful if you use functions overload that
     * does not take a WindowKind as parameter (for contrast, log and other window specific computation).
     *
     * \return WindowKind the current window type
     */
    inline static WindowKind get_current_window_type() { return GET_SETTING(CurrentWindow); }

    ContrastRange get_contrast_range(WindowKind kind) const;

    void set_contrast_range(ContrastRange range, WindowKind kind) const;

#pragma endregion

  public:
#pragma region Log

    /*! \brief Returns whether the log scale is enabled on the specified window kind (or the current window if not
     * specified).
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return bool true if the log scale is enabled
     */
    inline bool get_log_enabled(WindowKind kind = get_current_window_type()) const
    {
        return get_window(kind).log_enabled;
    }

    /*! \brief Enables or Disables log scale on the specified window kind (or the current window if not specified).
     *
     * \param[in] value true: enable, false: disable
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, OK otherwise
     */
    ApiCode set_log_enabled(bool value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Contrast Range

    /*! \brief Returns the contrast min value of the specified window kind (or the current window if not specified).
     *
     * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return float the contrast min value
     */
    float get_contrast_min(WindowKind kind = get_current_window_type()) const;

    /*! \brief Sets the contrast min value of the specified window kind (or the current window if not specified).
     *
     * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
     *
     * \param[in] value the new contrast min value
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if contrast is not enabled, OK otherwise
     */
    ApiCode set_contrast_min(float value, WindowKind kind = get_current_window_type()) const;

    /*! \brief Returns the contrast max value of the specified window kind (or the current window if not specified).
     *
     * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return float the contrast max value
     */
    float get_contrast_max(WindowKind kind = get_current_window_type()) const;

    /*! \brief Sets the contrast max value of the specified window kind (or the current window if not specified).
     *
     * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
     *
     * \param[in] value the new contrast max value
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if contrast is not enabled, OK otherwise
     */
    ApiCode set_contrast_max(float value, WindowKind kind = get_current_window_type()) const;

    /*! \brief Updates the contrast of the specified window kind (or the current window if not specified).
     *
     * The following formula is used to calculate the contrast: `px = (65535 / (max - min)) * (px - min)`
     *
     * \param[in] min the new contrast min value
     * \param[in] max the new contrast max value
     * \param[in] kind the window kind or the current window if not specified
     *
     * \warning This function is for internal use only, use set_contrast_min and set_contrast_max instead.
     */
    void update_contrast(float min, float max, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Contrast Enabled

    /*! \brief Returns whether the contrast is enabled on the specified window kind (or the current window if not
     * specified).
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return bool true if the contrast is enabled
     */
    inline bool get_contrast_enabled(WindowKind kind = get_current_window_type()) const
    {
        return get_window(kind).contrast.enabled;
    }

    /*! \brief Enables or Disables the contrast on the specified window kind (or the current window if not specified).
     *
     * \param[in] value true: enable, false: disable
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, OK otherwise
     */
    ApiCode set_contrast_enabled(bool value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Contrast Auto Refresh

    /*! \brief Returns whether the contrast auto refresh is enabled on the specified window kind (or the current window
     * if not specified).
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return bool true if the contrast auto refresh is enabled
     */
    inline bool get_contrast_auto_refresh(WindowKind kind = get_current_window_type()) const
    {
        return get_window(kind).contrast.auto_refresh;
    }

    /*! \brief Enables or Disables the contrast auto refresh on the specified window kind (or the current window
     * if not specified).
     *
     * \param[in] value true: enable, false: disable
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if contrast is not enabled, OK otherwise
     */
    ApiCode set_contrast_auto_refresh(bool value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Contrast Invert

    /*! \brief Returns whether the contrast is inverted (min and max will be swaped) on the specified window kind (or
     * the current window if not specified).
     *
     * \param[in] kind the window kind or the current window if not specified
     * \return bool true if the contrast is inverted
     */
    bool get_contrast_invert(WindowKind kind = get_current_window_type()) const;

    /*! \brief Enables or Disables the contrast invert on the specified window kind (or the current window
     * if not specified).
     *
     * \param[in] value true: enable, false: disable
     * \param[in] kind the window kind or the current window if not specified
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if contrast is not enabled, OK otherwise
     */
    ApiCode set_contrast_invert(bool value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Contrast Adv.

    /*! \brief Returns the contrast lower threshold pourcentage. This setting is used to determine which percentile will
     * be used for the min contrast value when computing the auto contrast. Is in range [0, 100].
     *
     * \return float the contrast lower threshold
     */
    inline float get_contrast_lower_threshold() const { return GET_SETTING(ContrastLowerThreshold); }

    /*! \brief Sets the contrast lower threshold pourcentage. This setting is used to determine which percentile will be
     * used for the min contrast value when computing the auto contrast. Must be in range [0, 100].
     *
     * \param[in] value the new contrast lower threshold in range [0, 100]
     *
     * \return ApiCode NO_CHANGE if the value is the same, INVALID_VALUE if not in range [0., 100.], OK otherwise
     */
    ApiCode set_contrast_lower_threshold(float value) const;

    /*! \brief Returns the contrast upper threshold pourcentage. This setting is used to determine which percentile will
     * be used for the max contrast value when computing the auto contrast. Is in range [0, 100].
     *
     * \return float the contrast upper threshold
     */
    inline float get_contrast_upper_threshold() const { return GET_SETTING(ContrastUpperThreshold); }

    /*! \brief Sets the contrast upper threshold pourcentage. This setting is used to determine which percentile will be
     * used for the max contrast value when computing the auto contrast. Must be in range [0, 100].
     *
     * \param[in] value the new contrast upper threshold in range [0, 100]
     *
     * \return ApiCode NO_CHANGE if the value is the same, INVALID_VALUE if not in range [0., 100.], OK otherwise
     */
    ApiCode set_contrast_upper_threshold(float value) const;

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
    inline uint get_cuts_contrast_p_offset() const { return static_cast<uint>(GET_SETTING(CutsContrastPOffset)); }

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
     *
     * \return ApiCode NO_CHANGE if the value is the same, INVALID_VALUE if not in range [0, get_fd().width / 2], OK
     * otherwise
     */
    ApiCode set_cuts_contrast_p_offset(uint value) const;

#pragma endregion

#pragma region Reticle

    /*! \brief Returns whether the reticle display is enabled. The reticle display is a rect region where the contrast
     * is calculated on.
     *
     * \return bool true if the reticle display is enabled
     */
    inline bool get_reticle_display_enabled() const { return GET_SETTING(ReticleDisplayEnabled); }

    /*! \brief Enables or Disables the reticle display. The reticle display is a rect region where the contrast is
     * calculated on.
     *
     * \param[in] value true: enable, false: disable
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, OK otherwise
     */
    ApiCode set_reticle_display_enabled(bool value) const;

    /*! \brief Returns the reticle scale. The reticle scale is the size of the reticle display.
     *
     * \return float the reticle scale
     */
    inline float get_reticle_scale() const { return GET_SETTING(ReticleScale); }

    /*! \brief Sets the reticle scale. The reticle scale is the size of the reticle display.
     *
     * \param[in] value the new reticle scale
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if reticle display is not enabled or if not in range [0., 1.], OK otherwise
     */
    ApiCode set_reticle_scale(float value) const;

    /*! \brief Recompute the reticle zone based on the reticle scale and the input frame descriptor.
     *
     * \warning This function is for internal use only.
     */
    void update_reticle_scale() const;

#pragma endregion
};

} // namespace holovibes::api