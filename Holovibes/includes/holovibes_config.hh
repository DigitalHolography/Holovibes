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
#define __PATCH_JSON_DIR__ "patch"
#ifndef _DEBUG
#define __BENCHMARK_FOLDER__ "benchmark"
#endif
const static std::string compute_settings_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = (__CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (__CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string patch_dirpath = (__CONFIG_FOLDER__ / __PATCH_JSON_DIR__).string();
#ifndef _DEBUG
const static std::string benchmark_dirpath = (__CONFIG_FOLDER__ / __BENCHMARK_FOLDER__).string();
#endif
} // namespace holovibes::settings
