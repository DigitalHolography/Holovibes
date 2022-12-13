/*! \file
 *
 * \brief Enum for kind of window
 */
#pragma once

#include "json_macro.hh"

namespace holovibes
{
/*! \enum WindowKind
 *
 * \brief Represents the kind of slice displayed by the window
 */
enum class WindowKind : int
{
    ViewXY = 0,   /*!< Main view */
    ViewXZ,       /*!< view slice */
    ViewYZ,       /*!< YZ view slice */
    ViewFilter2D, /*!< ViewFilter2D view */
};

// clang-format off
SERIALIZE_JSON_ENUM(WindowKind, {
    { WindowKind::ViewXY, "ViewXY"},
    { WindowKind::ViewXZ, "ViewXZ"},
    { WindowKind::ViewYZ, "ViewYZ"},
    { WindowKind::ViewFilter2D, "ViewFilter2D"}
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const WindowKind& value) { return os << json{value}; }

} // namespace holovibes
