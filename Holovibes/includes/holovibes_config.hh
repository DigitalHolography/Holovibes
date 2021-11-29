/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <string>
#include <filesystem>

namespace holovibes::ini
{
#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __GUI_CONFIG_FILENAME__ "user_settings.ini"
#define __CAMERAS_CONFIG_FOLDER__ "cameras_config"

#define __CONFIG_FOLDER__ (std::filesystem::path(getenv("AppData")) / __APPNAME__ / __HOLOVIBES_VERSION__)

const static std::string default_compute_config_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string global_config_filepath = (__CONFIG_FOLDER__ / __GUI_CONFIG_FILENAME__).string();
const static std::string camera_config_folderpath = (__CONFIG_FOLDER__ / __CAMERAS_CONFIG_FOLDER__).string();

} // namespace holovibes::ini