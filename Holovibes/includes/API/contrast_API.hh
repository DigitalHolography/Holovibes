#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

float get_contrast_min()
{
    return api::get_current_window().log_scale_slice_enabled ? api::get_current_window().contrast_min
                                                             : log10(api::get_current_window().contrast_min);
}

float get_contrast_max()
{
    return api::get_current_window().log_scale_slice_enabled ? api::get_current_window().contrast_max
                                                             : log10(api::get_current_window().contrast_max);
}

bool get_contrast_invert_enabled() { return api::get_current_window().contrast_invert; }

void set_auto_contrast_all();

} // namespace holovibes::api
