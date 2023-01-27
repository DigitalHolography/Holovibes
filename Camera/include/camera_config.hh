/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */

#pragma once

#include <string>
#include <filesystem>

#include "holovibes_config.hh"

namespace holovibes::settings
{

// Very ugly to redefine the version in the "camera" folder but we have no choice if we want to define it only once.
#define __HOLOVIBES_VERSION__ "11.6.0"
#define __APPNAME__ "Holovibes"

#define __CAMERAS_CONFIG_FOLDER__ "cameras_config"

#define __CAMERAS_CONFIG_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CAMERAS_CONFIG_FOLDER__)
} // namespace holovibes::settings
