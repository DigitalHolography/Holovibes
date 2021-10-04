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

namespace holovibes
{
namespace ini
{
static std::string global_ini_path = "";
std::string get_global_ini_path();

void load_ini(ComputeDescriptor& cd, const std::string& ini_path);
void load_ini(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd);
void save_ini(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd);
} // namespace ini
} // namespace holovibes
