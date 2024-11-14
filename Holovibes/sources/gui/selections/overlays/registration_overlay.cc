#include "API.hh"
#include "registration_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
RegistrationOverlay::RegistrationOverlay(BasicOpenGLWindow* parent)
    : CircOverlay(Registration, parent, 80)
{
    LOG_FUNC();

    display_ = true;
    color_ = {1.f, 0.f, 0.f};
}

void RegistrationOverlay::setBuffer()
{
    const float scale = api::get_registration_zone();

    scale_.x = scale;
    scale_.y = scale;
}
} // namespace holovibes::gui
