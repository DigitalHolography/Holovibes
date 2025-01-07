/*! \file
 *
 * \brief Declaration of start cli function
 */
#pragma once

#include "holovibes.hh"
#include "options_parser.hh"

namespace cli
{
#define DEFAULT_CLI_FPS INT_MAX

/*! \brief Start the command-line interface.
 *
 * \param[in] holovibes The holovibes object.
 * \param[in] opts The options descriptor.
 * \return The exit code.
 */
int start_cli(holovibes::Holovibes& holovibes, const holovibes::OptionsDescriptor& opts);
} // namespace cli
