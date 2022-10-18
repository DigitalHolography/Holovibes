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
#include "record_API.hh"
#include "display_API.hh"
#include "cuts_API.hh"
#include "contrast_API.hh"

namespace holovibes::api
{

void check_q_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (get_q_accu_level() > upper_bound)
        api::set_q_accu_level(upper_bound);

    upper_bound -= get_q_accu_level();

    if (upper_bound >= 0 && get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

} // namespace holovibes::api
