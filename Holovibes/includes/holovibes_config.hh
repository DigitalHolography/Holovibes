/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include <string>
#include <filesystem>

namespace holovibes::settings
{
#define __HOLOVIBES_VERSION__ "11.2.0"
#define __APPNAME__ "Holovibes"

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::path(getenv("AppData")) / __APPNAME__)

#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__ / __HOLOVIBES_VERSION__)


#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"
#define __PATCH_JSON_DIR__ "patch"

const static std::string compute_settings_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = (__CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (__CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string patch_dirpath = (__CONFIG_FOLDER__ / __PATCH_JSON_DIR__).string();
} // namespace holovibes::settings
