/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include "camera_config.hh"

namespace holovibes::settings
{
#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __GUI_CONFIG_FILENAME__ "user_settings.ini"
#define __EVERYTHING_LOG__ "everything.log"
#define __LASTEST_READABLE_LOG__ "latest_readable.log"

const static std::string default_compute_config_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string global_config_filepath = (__CONFIG_FOLDER__ / __GUI_CONFIG_FILENAME__).string();
const static std::string everything_log_path = (__CONFIG_FOLDER__ / __EVERYTHING_LOG__).string();
const static std::string latest_readable_path = (__CONFIG_FOLDER__ / __LASTEST_READABLE_LOG__).string();
} // namespace holovibes::settings
