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

void set_auto_contrast_all();

bool set_auto_contrast();

void set_contrast_min(const double value);
void set_contrast_max(const double value);

} // namespace holovibes::api
