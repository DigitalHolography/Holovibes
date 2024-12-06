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
    inline ViewXYZ get_window_xyz(WindowKind kind)
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
    inline void set_window_xyz(WindowKind kind, ViewXYZ value)
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

#pragma endregion

#pragma region Geometry Settings

    /*! \brief Returns whether the horizontal flip is activated on the specified window
     *
     * \param[in] kind the kind of window
     *
     * \return bool true if the horizontal flip is activated
     */
    inline bool get_horizontal_flip(WindowKind kind)
    {
        NOT_FILTER2D_R(kind, "horizontal flip", false);
        return get_window_xyz(kind).horizontal_flip;
    }

    /*! \brief Returns whether the horizontal flip is activated on the current window
     *
     * \return bool true if the horizontal flip is activated
     */
    bool get_horizontal_flip();

    /*! \brief Sets the horizontal flip state of the specified window
     *
     * \param[in] kind the kind of window
     * \param[in] value the new value of the horizontal flip state
     */
    void set_horizontal_flip(WindowKind kind, bool value);

    /*! \brief Sets the horizontal flip state of the current window
     *
     * \param[in] value the new value of the horizontal flip state
     */
    void set_horizontal_flip(bool value);

    /*! \brief Returns the rotation of the specified window. The rotation is in degrees and is applied anti-clockwise.
     * Only 0, 90, 180 and 270 degrees are supported.
     *
     * \param[in] kind the kind of window
     *
     * \return float the rotation of the window
     */
    inline float get_rotation(WindowKind kind)
    {
        NOT_FILTER2D_R(kind, "rotation", 0.0f);
        return get_window_xyz(kind).rotation;
    }

    /*! \brief Returns the rotation of the current window. The rotation is in degrees and is applied anti-clockwise.
     * Only 0, 90, 180 and 270 degrees are supported.
     *
     * \return float the rotation of the window
     */
    float get_rotation();

    /*! \brief Sets the rotation of the specified window. The rotation is in degrees and is applied anti-clockwise. Only
     * 0, 90, 180 and 270 degrees are supported.
     *
     * \param[in] kind the kind of window
     * \param[in] value the new rotation of the window either: 0, 90, 180 or 270 degrees.
     */
    void set_rotation(WindowKind kind, float value);

    /*! \brief Sets the rotation of the current window. The rotation is in degrees and is applied anti-clockwise. Only
     * 0, 90, 180 and 270 degrees are supported.
     *
     * \param[in] value the new rotation of the window either: 0, 90, 180 or 270 degrees.
     */
    void set_rotation(float value);

#pragma endregion

#pragma region Accumulation

    /*! \brief Returns the size of the accumulation window for the specified window. The accumulation is a post
     * processing step that takes `get_accumulation_level` images as input and average them to output a single image.
     *
     * High values of accumulation level will reduce the noise in the image but will also reduce the frame rate and
     * increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \param[in] kind the kind of window
     *
     * \return uint the size of the accumulation window.
     */
    inline uint get_accumulation_level(WindowKind kind)
    {
        NOT_FILTER2D_R(kind, "accumulation", 30);
        return get_window_xyz(kind).output_image_accumulation;
    }

    /*! \brief Returns the size of the accumulation window for the current window. The accumulation is a post processing
     * step that takes `get_accumulation_level` images as input and average them to output a single image.
     *
     * High values of accumulation level will reduce the noise in the image but will also reduce the frame rate and
     * increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \return uint the size of the accumulation window.
     */
    uint get_accumulation_level();

    /*! \brief Sets the size of the accumulation window for the specified window. Must be greater than 0.
     *
     * The accumulation is a post processing step that takes `get_accumulation_level` images as input and average them
     * to output a single image. High values of accumulation level will reduce the noise in the image but will also
     * reduce the frame rate and increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \param[in] kind the kind of window
     * \param[in] value the new size of the accumulation window.
     */
    void set_accumulation_level(WindowKind kind, uint value);

    /*! \brief Sets the size of the accumulation window for the current window. Must be greater than 0.
     *
     * The accumulation is a post processing step that takes `get_accumulation_level` images as input and average them
     * to output a single image. High values of accumulation level will reduce the noise in the image but will also
     * reduce the frame rate and increase motion artifacts. An accumulation level of 1 will disable the accumulation.
     *
     * \param[in] value the new size of the accumulation window, greater than 0.
     */
    void set_accumulation_level(uint value);

#pragma endregion

#pragma region Raw Bitshift

    /*! \brief Returns the value of raw bit shift. When raw data is displayed, pixels will be equal to: `px = px *
     * 2^RawBitshift`
     *
     * \return uint the value of the raw bit shift.
     */
    inline unsigned int get_raw_bitshift() { return static_cast<unsigned int>(GET_SETTING(RawBitshift)); }

    /*! \brief Sets value of raw bit shift. When raw data is displayed, pixels will be equal to: `px = px *
     * 2^RawBitshift`
     *
     * \param[in] value the new value of the raw bit shift.
     */
    inline void set_raw_bitshift(unsigned int value) { UPDATE_SETTING(RawBitshift, value); }

#pragma endregion

#pragma region Enabled

    /*! \brief Returns whether the specified window is enabled or not
     *
     * \param[in] kind the kind of window
     *
     * \return bool true if the window is enabled
     */
    inline bool get_enabled(WindowKind kind)
    {
        NOT_FILTER2D_R(kind, "enabled", false);
        return get_window_xyz(kind).enabled;
    }

    /*! \brief Returns whether the current window is enabled or not
     *
     * \return bool true if the window is enabled
     */
    bool get_enabled();

    /*! \brief Sets the enabled state of the specified window
     *
     * \param[in] kind the kind of window
     * \param[in] value the new value of the enabled state
     */
    inline void set_enabled(WindowKind kind, bool value)
    {
        NOT_FILTER2D(kind, "enabled");
        auto window = get_window_xyz(kind);
        window.enabled = value;
        set_window_xyz(kind, window);
    }

    /*! \brief Sets the enabled state of the current window
     *
     * \param[in] value the new value of the enabled state
     */
    void set_enabled(bool value);

#pragma endregion
};

} // namespace holovibes::api