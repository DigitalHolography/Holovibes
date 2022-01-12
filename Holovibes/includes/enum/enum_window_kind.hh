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
    XYview = 0, /*!< Main view */
    XZview,     /*!< view slice */
    YZview,     /*!< YZ view slice */
    Filter2D,   /*!< Filter2D view */
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(WindowKind)

} // namespace holovibes
