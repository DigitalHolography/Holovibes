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

#ifndef __CUSTOM_CONFIG_FOLDER__
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__ / __HOLOVIBES_VERSION__)
#else
#define __CONFIG_FOLDER__ (std::filesystem::path(__CUSTOM_CONFIG_FOLDER__))
#endif

#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"
#define __DUMP_FOLDER__ "dump"


const static std::string compute_settings_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = (__CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (__CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string dump_filepath = (__CONFIG_FOLDER__ / __DUMP_FOLDER__).string();
} // namespace holovibes::settings
