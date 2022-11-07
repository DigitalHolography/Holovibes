#include "API.hh"

namespace holovibes::api
{

void change_angle()
{
    double rot = api::get_current_window_as_view_xyz().rot;
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;
    api::change_current_window_as_view_xyz()->rot = new_rot;
}

void rotateTexture()
{
    change_angle();

    if (api::get_current_window_kind() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(api::get_view_xy().rot);
    else if (UserInterfaceDescriptor::instance().sliceXZ && api::get_current_window_kind() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(api::get_view_xz().rot);
    else if (UserInterfaceDescriptor::instance().sliceYZ && api::get_current_window_kind() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(api::get_view_yz().rot);
}

void change_flip()
{
    api::change_current_window_as_view_xyz()->flip_enabled = !api::get_current_window_as_view_xyz().flip_enabled;
}

void flipTexture()
{
    change_flip();

    if (api::get_current_window_kind() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(api::get_view_xy().flip_enabled);
    else if (UserInterfaceDescriptor::instance().sliceXZ && api::get_current_window_kind() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(api::get_view_xz().flip_enabled);
    else if (UserInterfaceDescriptor::instance().sliceYZ && api::get_current_window_kind() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(api::get_view_yz().flip_enabled);
}
} // namespace holovibes::api
