/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "logger.hh"
#include "compute_descriptor.hh"
#include "tools.hh"

namespace holovibes::ini
{

#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.ini"
#define __GUI_CONFIG_FILENAME__ "user_settings.ini"

#define __CONFIG_FOLDER__ (std::filesystem::path(getenv("AppData")) / __APPNAME__ / __HOLOVIBES_VERSION__)

const static std::string default_compute_config_filepath = (__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string global_config_filepath = (__CONFIG_FOLDER__ / __GUI_CONFIG_FILENAME__).string();

void load_compute_settings(ComputeDescriptor& cd, const std::string& ini_path);
void save_compute_settings(const ComputeDescriptor& cd, const std::string& ini_path);
} // namespace holovibes::ini
