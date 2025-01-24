/*! \file
 *
 * \brief Declaration of start cli function
 */
#pragma once

#include "API.hh"
#include "options_parser.hh"

namespace cli
{
#define DEFAULT_CLI_FPS INT_MAX

/*! \brief Start the command-line interface.
 *
 * \param[in] api The holovibes API.
 * \param[in] opts The options descriptor.
 * \return The exit code.
 */
int start_cli(holovibes::api::Api& api, const holovibes::OptionsDescriptor& opts);

} // namespace cli
