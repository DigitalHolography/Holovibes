#pragma once

#include "API_detail.hh"
#include "display_API.hh"

namespace holovibes::api
{

inline float get_contrast_min()
{
    return api::get_current_window().log_enabled ? api::get_current_window().contrast.min
                                                 : log10(api::get_current_window().contrast.min);
}

inline float get_contrast_max()
{
    return api::get_current_window().log_enabled ? api::get_current_window().contrast.max
                                                 : log10(api::get_current_window().contrast.max);
}

void set_auto_contrast_all();

bool set_auto_contrast();

void set_current_window_contrast_min(const float value);
void set_current_window_contrast_max(const float value);

} // namespace holovibes::api
