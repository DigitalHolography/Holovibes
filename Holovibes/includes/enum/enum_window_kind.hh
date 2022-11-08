/*! \file
 *
 * \brief Enum for kind of window
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum WindowKind
 *
 * \brief Represents the kind of slice displayed by the window
 */
enum class WindowKind
{
    XYview = 0,   /*!< Main view */
    XZview,       /*!< view slice */
    YZview,       /*!< YZ view slice */
    ViewFilter2D, /*!< ViewFilter2D view */
};

// clang-format off
SERIALIZE_JSON_ENUM(WindowKind, {
    { WindowKind::XYview, "XYview"},
    { WindowKind::XZview, "XZview"},
    { WindowKind::YZview, "YZview"},
    { WindowKind::ViewFilter2D, "ViewFilter2D"}
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const WindowKind& value) { return os << json{value}; }

} // namespace holovibes
