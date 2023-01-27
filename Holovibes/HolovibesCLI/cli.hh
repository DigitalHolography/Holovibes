/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "holovibes.hh"
#include "options_parser.hh"
#include "CUDA_API.hh"

namespace holovibes::cli
{

#define DEFAULT_CLI_FPS INT_MAX

void start_cli(const holovibes::OptionsDescriptor& opts);

} // namespace holovibes::cli
