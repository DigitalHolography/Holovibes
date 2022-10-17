#include "API.hh"

namespace holovibes::api
{

void change_angle()
{
    double rot = api::get_current_window_as_view_xyz().get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;
    api::change_current_window_as_view_xyz()->set_rotation(new_rot);
}

void rotateTexture()
{
    change_angle();

    if (api::get_current_window_kind() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(api::get_view_xy().get_rotation());
    else if (UserInterfaceDescriptor::instance().sliceXZ && api::get_current_window_kind() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(api::get_view_xz().get_rotation());
    else if (UserInterfaceDescriptor::instance().sliceYZ && api::get_current_window_kind() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(api::get_view_yz().get_rotation());
}

void change_flip()
{
    api::change_current_window_as_view_xyz()->set_flip_enabled(
        !api::get_current_window_as_view_xyz().get_flip_enabled());
}

void flipTexture()
{
    change_flip();

    if (api::get_current_window_kind() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(api::get_view_xy().get_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceXZ && api::get_current_window_kind() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(api::get_view_xz().get_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceYZ && api::get_current_window_kind() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(api::get_view_yz().get_flip_enabled());
}
} // namespace holovibes::api
