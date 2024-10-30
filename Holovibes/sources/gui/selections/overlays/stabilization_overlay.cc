#include "API.hh"
#include "stabilization_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
StabilizationOverlay::StabilizationOverlay(BasicOpenGLWindow* parent)
    : CircOverlay(Stabilization, parent, 80)
{
    LOG_FUNC();

    display_ = true;
    color_ = {0.f, 0.f, 1.f};
}

void StabilizationOverlay::setBuffer()
{
    const float scale = api::get_reticle_scale(); // TODO change this by the mask radius for the stabilization
    const float w = parent_->size().width();
    const float h = parent_->size().height();

    units::ConversionData convert(parent_);
    auto mid = units::PointWindow(convert, w / 2, h / 2);

    zone_ = units::RectWindow(mid, mid);
    radius_ = scale;

    CircOverlay::setBuffer();
}
} // namespace holovibes::gui