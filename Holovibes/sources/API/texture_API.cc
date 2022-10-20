#include "API.hh"
#include "view_API.hh"
#include "display_API.hh"

namespace holovibes::api
{

void change_angle()
{
    double rot = GSH::instance().get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    GSH::instance().set_rotation(new_rot);
}

void rotateTexture()
{
    change_angle();

    if (GSH::instance().get_value<CurrentWindowKind>() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(GSH::instance().get_xy_rot());
    else if (UserInterfaceDescriptor::instance().sliceXZ &&
             GSH::instance().get_value<CurrentWindowKind>() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(GSH::instance().get_xz_rot());
    else if (UserInterfaceDescriptor::instance().sliceYZ &&
             GSH::instance().get_value<CurrentWindowKind>() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(GSH::instance().get_yz_rot());
}

void change_flip() { GSH::instance().set_flip_enabled(!GSH::instance().get_flip_enabled()); }

void flipTexture()
{
    change_flip();

    if (GSH::instance().get_value<CurrentWindowKind>() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(GSH::instance().get_xy_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceXZ &&
             GSH::instance().get_value<CurrentWindowKind>() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(GSH::instance().get_xz_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceYZ &&
             GSH::instance().get_value<CurrentWindowKind>() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(GSH::instance().get_yz_flip_enabled());
}
} // namespace holovibes::api
