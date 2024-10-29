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
    alpha_ = 1.0f;
    filled_ = false;
}

void ReticleOverlay::setBuffer()
{
    const float scale = api::get_reticle_scale();
    const float w = parent_->size().width();
    const float h = parent_->size().height();

    const float w_2 = w / 2;
    const float h_2 = h / 2;

    units::ConversionData convert(parent_);
    auto top_left = units::PointWindow(convert, w_2 - w_2 * scale, h_2 - h_2 * scale);
    auto bottom_right = units::PointWindow(convert, w_2 + w_2 * scale, h_2 + h_2 * scale);

    zone_ = units::RectWindow(top_left, bottom_right);
    api::set_reticle_zone(zone_);

    RectOverlay::setBuffer();
}
} // namespace holovibes::gui
