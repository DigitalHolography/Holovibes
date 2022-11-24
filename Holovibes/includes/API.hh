/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "API_detail.hh"
#include "advanced_API.hh"
#include "compute_API.hh"
#include "composite_API.hh"
#include "export_API.hh"
#include "import_API.hh"
#include "view_API.hh"
#include "zone_API.hh"
#include "record_API.hh"
#include "display_API.hh"
#include "cuts_API.hh"
#include "contrast_API.hh"
#include "pipe_API.hh"
#include "credits_API.hh"
#include "holovibes_API.hh"
#include "camera_API.hh"
#include "compute_settings_API.hh"
#include "compute_settings_struct.hh"
#include "unknown_API.hh"
#include "CUDA_API.hh"
#include "options_parser.hh"

namespace holovibes::api
{
inline void print_version() { std::cout << "Holovibes " << __HOLOVIBES_VERSION__ << std::endl; }

inline void print_help(OptionsParser& parser)
{
    print_version();
    std::cout << "Usage: ./Holovibes.exe [OPTIONS]" << std::endl;
    std::cout << parser.get_opts_desc();
}
} // namespace holovibes::api
