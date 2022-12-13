/*! \file
 *
 * \brief Enum for the different color themes
 */
#pragma once

#include "json_macro.hh"

namespace holovibes
{
/*! \enum Theme
 *
 * \brief The color theme of the GUI
 */
enum class Theme
{
    Classic = 0,
    Dark
};

// clang-format off
SERIALIZE_JSON_ENUM(Theme, {
    {Theme::Classic, "CLASSIC"},
    {Theme::Dark, "DARK"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const Theme& value) { return os << json{value}; }

} // namespace holovibes