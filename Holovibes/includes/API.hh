/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "API_detail.hh"
#include "advanced_cache_API.hh"
#include "compute_cache_API.hh"
#include "compute_settings_struct.hh"
#include "composite_cache_API.hh"
#include "export_cache_API.hh"
#include "import_cache_API.hh"
#include "file_read_cache_API.hh"
#include "filter2d_cache_API.hh"
#include "view_cache_API.hh"
#include "zone_cache_API.hh"

namespace holovibes::api
{

const View_Window& get_current_window() const { return GSH::instance().get_current_window(); }

} // namespace holovibes::api
