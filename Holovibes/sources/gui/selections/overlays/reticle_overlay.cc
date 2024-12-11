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
    const float w_2 = parent_->size().width() / 2;
    const float h_2 = parent_->size().height() / 2;

    units::ConversionData convert(parent_);
    auto top_left = units::PointWindow(convert, w_2 - w_2 * scale, h_2 - h_2 * scale);
    auto bottom_right = units::PointWindow(convert, w_2 + w_2 * scale, h_2 + h_2 * scale);

    zone_ = units::RectWindow(top_left, bottom_right);
    API.contrast.set_reticle_zone(zone_);

    scale_.x = scale;
    scale_.y = scale;
}
} // namespace holovibes::gui
