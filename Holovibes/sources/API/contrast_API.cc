#include "API.hh"

namespace holovibes::api
{

void set_auto_contrast_all()
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    get_compute_pipe().request_autocontrast(WindowKind::XYview);
    if (api::get_cuts_view_enabled())
    {
        get_compute_pipe().request_autocontrast(WindowKind::XZview);
        get_compute_pipe().request_autocontrast(WindowKind::YZview);
    }
    if (api::get_filter2d_view_enabled())
        get_compute_pipe().request_autocontrast(WindowKind::Filter2D);

    pipe_refresh();
}
} // namespace holovibes::api
