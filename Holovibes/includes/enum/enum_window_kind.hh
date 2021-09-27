/*! \file
 *
 * \brief Enum for kind of window
 */
#pragma once

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
} // namespace holovibes