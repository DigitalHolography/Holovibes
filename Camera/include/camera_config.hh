/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */

#pragma once

#include <string>
#include <filesystem>

namespace holovibes::settings
{
// Very ugly to redefine the version in the "camera" folder but we have no choice if we want to define it only once.
#define __HOLOVIBES_VERSION__ "11.0.0"
#define __APPNAME__ "Holovibes"

#define __CAMERAS_CONFIG_FOLDER__ "cameras_config"

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::path(getenv("AppData")) / __APPNAME__)
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__ / __HOLOVIBES_VERSION__)

// #define __CAMERAS_CONFIG_FOLDER_PATH__                                                                                 \
//     std::filesystem::path("C:/Users/Karachayevsk/Desktop/Stagiaires-2022/david.chemaly/Holovibes/Camera/configs")

#define __CAMERAS_CONFIG_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CAMERAS_CONFIG_FOLDER__)

} // namespace holovibes::settings
