/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include "camera_config.hh"

namespace holovibes::settings
{
#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"

const static std::string compute_settings_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = (__CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (__CONFIG_FOLDER__ / __LOGS_DIR__).string();
} // namespace holovibes::settings
