/*! \file
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

#pragma region Internal

WindowKind get_current_window_type();

#define NOT_FILTER2D_R(kind, text, r_value)                                                                            \
    if (kind == WindowKind::Filter2D)                                                                                  \
    {                                                                                                                  \
        LOG_WARN("Filter2D window has no {} setting", text);                                                           \
        return r_value;                                                                                                \
    }

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

#define NOT_FILTER2D(kind, text)                                                                                       \
    if (kind == WindowKind::Filter2D)                                                                                  \
    {                                                                                                                  \
        LOG_WARN("Filter2D window has no {} setting", text);                                                           \
        return;                                                                                                        \
    }

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

inline bool get_horizontal_flip(WindowKind kind)
{
    NOT_FILTER2D_R(kind, "horizontal flip", false);
    return get_window_xyz(kind).horizontal_flip;
}

inline float get_rotation(WindowKind kind)
{
    NOT_FILTER2D_R(kind, "rotation", 0.0f);
    return get_window_xyz(kind).rotation;
}

inline bool get_horizontal_flip() { return get_horizontal_flip(get_current_window_type()); }
inline float get_rotation() { return get_rotation(get_current_window_type()); }

void set_horizontal_flip(WindowKind kind, bool value);
void set_rotation(WindowKind kind, float value);

void set_horizontal_flip(bool value);
void set_rotation(float value);

#pragma endregion

#pragma region Accumulation

inline uint get_accumulation_level(WindowKind kind)
{
    NOT_FILTER2D_R(kind, "accumulation", 30);
    return get_window_xyz(kind).output_image_accumulation;
}

inline uint get_accumulation_level() { return get_accumulation_level(get_current_window_type()); }

void set_accumulation_level(WindowKind kind, uint value);
void set_accumulation_level(uint value);

#pragma endregion

#pragma region Raw Bitshift

/*! \brief Set value of raw bit shift
 *
 * \param value to set
 */
inline void set_raw_bitshift(unsigned int value) { UPDATE_SETTING(RawBitshift, value); }

/*!
 * \brief Gets the raw bit shift
 *
 * \return uint the raw bit shift
 */
inline unsigned int get_raw_bitshift() { return static_cast<unsigned int>(GET_SETTING(RawBitshift)); }

#pragma endregion

inline bool get_enabled(WindowKind kind)
{
    NOT_FILTER2D_R(kind, "enabled", false);
    return get_window_xyz(kind).enabled;
}

inline bool get_enabled() { return get_enabled(get_current_window_type()); }

inline void set_enabled(WindowKind kind, bool value)
{
    NOT_FILTER2D(kind, "enabled");
    auto window = get_window_xyz(kind);
    window.enabled = value;
    set_window_xyz(kind, window);
}

inline void set_enabled(bool value) { set_enabled(get_current_window_type(), value); }

inline units::RectFd get_zoomed_zone() { return GET_SETTING(ZoomedZone); }
inline void set_zoomed_zone(const units::RectFd& rect) { UPDATE_SETTING(ZoomedZone, rect); }

} // namespace holovibes::api