#include "API.hh"
#include "signal_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
SignalOverlay::SignalOverlay(BasicOpenGLWindow* parent)
    : FilledRectOverlay(KindOfOverlay::Signal, parent)
{
    color_ = {0.557f, 0.4f, 0.85f};
    alpha_ = 1.f;
    fill_alpha_ = 0.4f;
    filled_ = false;
}

void SignalOverlay::release(ushort frameSide)
{
    if (parent_->getKindOfView() == KindOfView::Hologram)
        api::set_signal_zone(zone_);
}
} // namespace holovibes::gui
