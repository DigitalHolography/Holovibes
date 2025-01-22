/*! \file windowpostprocess_api.hh
 *
 * \brief Regroup all functions used to interact with post processing operations done on all windows.
 *
 * Windows are XY, XZ, YZ and Filter2D.
 *
 * Operations are:
 * - image accumulation
 * - rotation
 * - flip
 */
#pragma once

#include "common_api.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

class WindowPostProcessApi : public IApi
{

  public:
    WindowPostProcessApi(const Api* api)
        : IApi(api)
    {
    }

  private:
#pragma region Internals
/*! \brief Test whether the current window is a Filter2D window. If it is, print a warning in the log and return
 * r_value.
 *
 * \param[in] kind the kind of window to test
 * \param[in] text the text to display in the log in case if Filter2D window
 * \param[in] r_value the value to return if the current window is a Filter2D window
 *
 * \return r_value if the current window is a Filter2D window
 */
#define NOT_FILTER2D_R(kind, text, r_value)                                                                            \
    if (kind == WindowKind::Filter2D)                                                                                  \
    {                                                                                                                  \
        LOG_WARN("Filter2D window has no {} setting", text);                                                           \
        return r_value;                                                                                                \
    }

    /*! \brief Get the current value of a setting for a window
     *
     * \param[in] kind the kind of window
     *
     * \return the value of the setting
     */
    inline ViewXYZ get_window_xyz(WindowKind kind) const
    {
        switch (kind)
        {
        case WindowKind::XYview:
            return GET_SETTING(XY);
        case WindowKind::XZview:
            return GET_SETTING(XZ);
        default:
            return GET_SETTING(YZ);
        }
    }

/*! \brief Test whether the current window is a Filter2D window. If it is, print a warning in the log and return.
 *
 * \param[in] kind the kind of window to test
 * \param[in] text the text to display in the log in case if Filter2D window
 */
#define NOT_FILTER2D(kind, text)                                                                                       \
    if (kind == WindowKind::Filter2D)                                                                                  \
    {                                                                                                                  \
        LOG_WARN("Filter2D window has no {} setting", text);                                                           \
        return;                                                                                                        \
    }

    /*! \brief Update the value of a setting for a window
     *
     * \param[in] kind the kind of window
     * \param[in] value the new value of the setting
     */
    inline void set_window_xyz(WindowKind kind, ViewXYZ value) const
    {
        switch (kind)
        {
        case WindowKind::XYview:
            UPDATE_SETTING(XY, value);
            break;
        case WindowKind::XZview:
            UPDATE_SETTING(XZ, value);
            break;
        case WindowKind::YZview:
            UPDATE_SETTING(YZ, value);
            break;
        default:
            break;
        }
    }

    /*! \brief Returns the type of the focused window. This setting is only useful if you use functions overload that
     * does not take a WindowKind as parameter (for contrast, log and other window specific computation).
     *
     * \return WindowKind the current window type
     */
    inline static WindowKind get_current_window_type() { return GET_SETTING(CurrentWindow); }

#pragma endregion

  public:
#pragma region Geometry Settings

    /*! \brief Returns whether the horizontal flip is activated on the specified window (or the current window if not
     * specified)
     *
     * \param[in] kind the kind of window or the current window if not specified
     *
     * \return bool true if the horizontal flip is activated
     */
    inline bool get_horizontal_flip(WindowKind kind = get_current_window_type()) const
    {
        NOT_FILTER2D_R(kind, "horizontal flip", false);
        return get_window_xyz(kind).horizontal_flip;
    }

    /*! \brief Sets the horizontal flip state of the specified window (or the current window if not specified)
     *
     * \param[in] value the new value of the horizontal flip state
     * \param[in] kind the kind of window or the current window if not specified
     */
    void set_horizontal_flip(bool value, WindowKind kind = get_current_window_type()) const;

    /*! \brief Returns the rotation of the specified window (or the current window if not specified). The rotation is in
     * degrees and is applied anti-clockwise. Only 0, 90, 180 and 270 degrees are supported.
     *
     * \param[in] kind the kind of window or the current window if not specified.
     *
     * \return float the rotation of the window
     */
    inline float get_rotation(WindowKind kind = get_current_window_type()) const
    {
        NOT_FILTER2D_R(kind, "rotation", 0.0f);
        return get_window_xyz(kind).rotation;
    }

    /*! \brief Sets the rotation of the specified window (or the current window if not specified). The rotation is in
     * degrees and is applied anti-clockwise. Only 0, 90, 180 and 270 degrees are supported.
     *
     * \param[in] value the new rotation of the window either: 0, 90, 180 or 270 degrees.
     * \param[in] kind the kind of window or the current window if not specified.
     */
    void set_rotation(float value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Accumulation

    /*! \brief Returns the size of the accumulation window for the specified window (or the current window if not
     * specified). The accumulation is a post processing step that takes `get_accumulation_level` images as input and
     * average them to output a single image.
     *
     * High values of accumulation level will reduce the noise in the image but will also reduce the frame rate and
     * increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \param[in] kind the kind of window or the current window if not specified.
     *
     * \return uint the size of the accumulation window.
     */
    inline uint get_accumulation_level(WindowKind kind = get_current_window_type()) const
    {
        NOT_FILTER2D_R(kind, "accumulation", 30);
        return get_window_xyz(kind).output_image_accumulation;
    }

    /*! \brief Sets the size of the accumulation window for the specified window (or the current window if not
     * specified). Must be greater than 0.
     *
     * The accumulation is a post processing step that takes `get_accumulation_level` images as input and average them
     * to output a single image. High values of accumulation level will reduce the noise in the image but will also
     * reduce the frame rate and increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \param[in] value the new size of the accumulation window.
     * \param[in] kind the kind of window or the current window if not specified.
     */
    void set_accumulation_level(uint value, WindowKind kind = get_current_window_type()) const;

#pragma endregion

#pragma region Raw Bitshift

    /*! \brief Returns the value of raw bit shift. When raw data is displayed, pixels will be equal to: `px = px *
     * 2^RawBitshift`
     *
     * \return uint the value of the raw bit shift.
     */
    inline unsigned int get_raw_bitshift() const { return static_cast<unsigned int>(GET_SETTING(RawBitshift)); }

    /*! \brief Sets value of raw bit shift. When raw data is displayed, pixels will be equal to: `px = px *
     * 2^RawBitshift`
     *
     * \param[in] value the new value of the raw bit shift.
     */
    inline void set_raw_bitshift(unsigned int value) const { UPDATE_SETTING(RawBitshift, value); }

#pragma endregion

#pragma region Enabled

    /*! \brief Returns whether the specified window (or the current window if not specified) is enabled or not
     *
     * \param[in] kind the kind of window or the current window if not specified.
     *
     * \return bool true if the window is enabled
     */
    inline bool get_enabled(WindowKind kind = get_current_window_type()) const
    {
        NOT_FILTER2D_R(kind, "enabled", false);
        return get_window_xyz(kind).enabled;
    }

    /*! \brief Sets the enabled state of the specified window (or the current window if not specified).
     *
     * \param[in] value the new value of the enabled state
     * \param[in] kind the kind of window or the current window if not specified.
     */
    inline void set_enabled(bool value, WindowKind kind = get_current_window_type()) const
    {
        NOT_FILTER2D(kind, "enabled");
        auto window = get_window_xyz(kind);
        window.enabled = value;
        set_window_xyz(kind, window);
    }

    friend class ContrastApi;

#pragma endregion
};

} // namespace holovibes::api