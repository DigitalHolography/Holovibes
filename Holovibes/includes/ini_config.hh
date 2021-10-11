/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "logger.hh"
#include "config.hh"
#include "compute_descriptor.hh"
#include "tools.hh"

namespace holovibes::ini
{

#define __CONFIG_FILENAME__ "holovibes.ini"

#define __CONFIG_FOLDER__ (std::filesystem::path(getenv("AppData")) / __APPNAME__ / __HOLOVIBES_VERSION__)

const static std::string default_config_filepath = (__CONFIG_FOLDER__ / __CONFIG_FILENAME__).string();

void load_ini(ComputeDescriptor& cd, const std::string& ini_path);
void save_ini(const ComputeDescriptor& cd, const std::string& ini_path);

} // namespace holovibes::ini
