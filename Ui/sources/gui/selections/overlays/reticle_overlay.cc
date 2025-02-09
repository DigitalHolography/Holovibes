#include "API.hh"
#include "reticle_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
ReticleOverlay::ReticleOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(Reticle, parent)
{
    LOG_FUNC();

    display_ = true;
}

void ReticleOverlay::setBuffer()
{
    const float scale = API.contrast.get_reticle_scale();

    zone_ = API.contrast.get_reticle_zone();

    scale_.x = scale;
    scale_.y = scale;
}
} // namespace holovibes::gui
