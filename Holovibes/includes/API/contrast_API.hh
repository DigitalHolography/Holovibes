#pragma once

#include "API_detail.hh"
#include "display_API.hh"
#include "view_cache_API.hh"

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

void set_auto_contrast_all()
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    auto pipe = get_compute_pipe();
    pipe->request_autocontrast(WindowKind::XYview);
    if (api::get_cuts_view_enabled())
    {
        pipe->request_autocontrast(WindowKind::XZview);
        pipe->request_autocontrast(WindowKind::YZview);
    }
    if (api::get_filter2d_view_enabled())
        pipe->request_autocontrast(WindowKind::Filter2D);

    pipe_refresh();
}

} // namespace holovibes::api
