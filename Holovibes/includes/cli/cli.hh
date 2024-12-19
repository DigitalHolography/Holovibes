/*! \file
 *
 * \brief declaration of start cli function
 */
#pragma once

#include "holovibes.hh"
#include "options_parser.hh"

namespace cli
{
#define DEFAULT_CLI_FPS INT_MAX

int start_cli(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts);
} // namespace cli
