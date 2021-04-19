/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include "config.hh"
#include "compute_descriptor.hh"

namespace holovibes
{
namespace ini
{
void load_ini(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd);
void save_ini(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd);
} // namespace ini
} // namespace holovibes
