#include "API.hh"
#include "reticle_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
ContrastReticleOverlay::ContrastReticleOverlay(BasicOpenGLWindow* parent)
    : CircOverlay(ContrastReticle, parent, 80)
{
    LOG_FUNC();

    display_ = true;
    color_ = {1.f, 0.f, 0.f};
}

void ContrastReticleOverlay::setBuffer()
{
    const float scale = API.contrast.get_contrast_reticle_scale();

    scale_.x = scale;
    scale_.y = scale;
}

ReticleOverlay::ReticleOverlay(BasicOpenGLWindow* parent)
    : CircOverlay(Reticle, parent, 80)
{
    LOG_FUNC();

    display_ = true;
    color_ = {0.f, 0.f, 1.f};
}

void ReticleOverlay::setBuffer()
{
    const float scale = API.contrast.get_reticle_scale();

    scale_.x = scale;
    scale_.y = scale;
}
} // namespace holovibes::gui
