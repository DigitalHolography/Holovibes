/*! \file
 *
 * \brief Enum for the different color themes
 */
#pragma once

#include <map>

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

static std::map<std::string, Theme> string_to_theme = {
    {"CLASSIC", Theme::Classic},
    {"DARK", Theme::Dark},
};

static std::map<Theme, std::string> theme_to_string = {
    {Theme::Classic, "CLASSIC"},
    {Theme::Dark, "DARK"},
};

} // namespace holovibes