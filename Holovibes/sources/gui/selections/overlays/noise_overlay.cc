#include "API.hh"
#include "noise_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
NoiseOverlay::NoiseOverlay(BasicOpenGLWindow* parent)
    : FilledRectOverlay(KindOfOverlay::Noise, parent)
{
    color_ = {0.f, 0.64f, 0.67f};
}

void NoiseOverlay::release(ushort frameSide)
{
    if (parent_->getKindOfView() == KindOfView::Hologram)
        API.record.set_noise_zone(zone_);
}
} // namespace holovibes::gui
