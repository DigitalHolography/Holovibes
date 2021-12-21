#include "API.hh"
#include "signal_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
SignalOverlay::SignalOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::Signal, parent)
{
    color_ = {0.557f, 0.4f, 0.85f};
}

void SignalOverlay::release(ushort frameSide)
{
    if (parent_->getKindOfView() == KindOfView::Hologram)
        api::set_signal_zone(zone_);
}
} // namespace gui
} // namespace holovibes
