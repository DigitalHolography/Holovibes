#pragma once

#include "API_detail.hh"
#include "display_API.hh"

namespace holovibes::api
{

inline float get_contrast_min()
{
    return api::get_current_window().log_scale_slice_enabled ? api::get_current_window().contrast_min
                                                             : log10(api::get_current_window().contrast_min);
}

inline float get_contrast_max()
{
    return api::get_current_window().log_scale_slice_enabled ? api::get_current_window().contrast_max
                                                             : log10(api::get_current_window().contrast_max);
}

void set_auto_contrast_all();

bool set_auto_contrast();

void set_current_window_contrast_min(const float value);
void set_current_window_contrast_max(const float value);

} // namespace holovibes::api
