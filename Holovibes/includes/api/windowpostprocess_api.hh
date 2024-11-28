// /*! \file
//  *
//  * \brief Regroup all functions used to interact with post processing operations done on all windows.
//  *
//  * Windows are XY, XZ, YZ and Filter2D.
//  *
//  * Operations are:
//  * - image accumulation
//  * - rotation
//  * - flip
//  */
// #pragma once

// #include "common_api.hh"
// #include "enum_window_kind.hh"

// namespace holovibes::api
// {

// #pragma region Internal

// #define NOT_FILTER2D_R(kind, text, r_value)                                                                            \
//     if (kind == WindowKind::Filter2D)                                                                                  \
//     {                                                                                                                  \
//         LOG_WARN("Filter2D window has no {} setting", text);                                                           \
//         return r_value;                                                                                                \
//     }

// inline ViewXYZ get_window_xyz(WindowKind kind)
// {
//     switch (kind)
//     {
//     case WindowKind::XYview:
//         return GET_SETTING(XY);
//     case WindowKind::XZview:
//         return GET_SETTING(XZ);
//     default:
//         return GET_SETTING(YZ);
//     }
// }

// #define NOT_FILTER2D(kind, text)                                                                                       \
//     if (kind == WindowKind::Filter2D)                                                                                  \
//     {                                                                                                                  \
//         LOG_WARN("Filter2D window has no {} setting", text);                                                           \
//         return;                                                                                                        \
//     }

// inline void set_window_xyz(WindowKind kind, ViewXYZ value)
// {
//     switch (kind)
//     {
//     case WindowKind::XYview:
//         set_xy(value);
//         break;
//     case WindowKind::XZview:
//         set_xz(value);
//         break;
//     case WindowKind::YZview:
//         set_yz(value);
//         break;
//     default:
//         break;
//     }
// }

// #pragma endregion

// #pragma region Geometry Settings

// inline bool get_horizontal_flip(WindowKind kind)
// {
//     NOT_FILTER2D_R(kind, "horizontal flip", false);
//     return get_window_xyz(kind).horizontal_flip;
// }

// inline float get_rotation(WindowKind kind)
// {
//     NOT_FILTER2D_R(kind, "rotation", 0.0f);
//     return get_window_xyz(kind).rotation;
// }

// inline bool get_horizontal_flip() { return get_horizontal_flip(get_current_window_type()); }
// inline float get_rotation() { return get_rotation(get_current_window_type()); }

// inline void set_horizontal_flip(WindowKind kind, bool value)
// {
//     NOT_FILTER2D(kind, "horizontal flip");
//     auto window = get_window_xyz(kind);
//     window.horizontal_flip = value;
//     set_window_xyz(kind, window);
// }

// inline void set_rotation(WindowKind kind, float value)
// {
//     NOT_FILTER2D(kind, "rotation");
//     auto window = get_window_xyz(kind);
//     window.rotation = value;
//     set_window_xyz(kind, window);
// }

// inline void set_horizontal_flip(bool value) { return set_horizontal_flip(get_current_window_type(), value); }
// inline void set_rotation(float value) { return set_rotation(get_current_window_type(), value); }

// #pragma endregion

// #pragma region Accumulation

// inline uint get_accumulation_level(WindowKind kind)
// {
//     NOT_FILTER2D_R(kind, "accumulation", 1);
//     return get_window_xyz(kind).output_image_accumulation;
// }

// inline uint get_accumulation_level() { return get_accumulation_level(get_current_window_type()); }

// inline void set_accumulation_level(WindowKind kind, uint value)
// {
//     NOT_FILTER2D(kind, "accumulation");
//     auto window = get_window_xyz(kind);
//     window.output_image_accumulation = value;
//     set_window_xyz(kind, window);
// }

// inline void set_accumulation_level(uint value) { return set_accumulation_level(get_current_window_type(), value); }

// #pragma endregion

// inline bool get_enabled(WindowKind kind)
// {
//     NOT_FILTER2D_R(kind, "enabled", false);
//     return get_window_xyz(kind).enabled;
// }

// inline bool get_enabled() { return get_enabled(get_current_window_type()); }

// inline void set_enabled(WindowKind kind, bool value)
// {
//     NOT_FILTER2D(kind, "enabled");
//     auto window = get_window_xyz(kind);
//     window.enabled = value;
//     set_window_xyz(kind, window);
// }

// inline void set_enabled(bool value) { return set_enabled(get_current_window_type(), value); }

// } // namespace holovibes::api