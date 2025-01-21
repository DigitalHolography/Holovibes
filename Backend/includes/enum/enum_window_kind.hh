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
    XYview,   /*!< Main view */
    XZview,   /*!< view slice */
    YZview,   /*!< YZ view slice */
    Filter2D, /*!< Filter2D view */
};

// clang-format off
SERIALIZE_JSON_ENUM(WindowKind, {
    { WindowKind::XYview, "XYview"},
    { WindowKind::XZview, "XZview"},
    { WindowKind::YZview, "YZview"},
    { WindowKind::Filter2D, "Filter2D"}
})
// clang-format on
} // namespace holovibes
