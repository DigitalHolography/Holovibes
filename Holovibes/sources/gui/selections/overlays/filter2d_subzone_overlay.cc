/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "filter2d_subzone_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
namespace gui
{
Filter2DSubZoneOverlay::Filter2DSubZoneOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::Filter2DSubZone, parent)
{
    color_ = {0.62f, 0.f, 1.f};
}

void Filter2DSubZoneOverlay::release(ushort frameSide)
{
    checkCorners();

    if (zone_.src() == zone_.dst())
        return;

    // handle Filter2D
    auto window = dynamic_cast<HoloWindow*>(parent_);
    if (window)
    {
        window->getCd()->setFilter2DSubZone(zone_);
        window->getPipe()->request_filter2D_roi_update();
        window->getPipe()->request_filter2D_roi_end();
    }

    parent_->getCd()->fft_shift_enabled = false;

    filter2d_overlay_->disable();

    active_ = false;
}

void Filter2DSubZoneOverlay::setFilter2dOverlay(
    std::shared_ptr<Filter2DOverlay> rhs)
{
    filter2d_overlay_ = rhs;
}

} // namespace gui
} // namespace holovibes
