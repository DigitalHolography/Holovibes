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

const std::map<WindowKind, std::string> window_kind_to_string{{
                                                                             WindowKind::XYview,
                                                                             "XYview",
                                                                         },
                                                                         {
                                                                             WindowKind::XZview,
                                                                             "XZview",
                                                                         },
                                                                         {
                                                                             WindowKind::YZview,
                                                                             "YZview",
                                                                         },
                                                                         {
                                                                             WindowKind::Filter2D,
                                                                             "Filter2D",
                                                                         }};


inline std::ostream& operator<<(std::ostream& os, WindowKind obj) {
    return os << window_kind_to_string.at(obj);
}
} // namespace holovibes
