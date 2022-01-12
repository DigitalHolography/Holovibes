/*! \file
 *
 * \brief Enum for the different color themes
 */
#pragma once

#include "all_struct.hh"

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

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(Theme)

} // namespace holovibes