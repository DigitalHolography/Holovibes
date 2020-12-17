/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "filter2d_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "HoloWindow.hh"

namespace holovibes
{
namespace gui
{
Filter2DOverlay::Filter2DOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::Filter2D, parent)
{
    color_ = {0.f, 0.62f, 1.f};
}

void Filter2DOverlay::release(ushort frameSide)
{
    checkCorners();

    if (zone_.src() == zone_.dst())
        return;

    // handle Filter2D
    auto window = dynamic_cast<HoloWindow*>(parent_);
    if (window)
    {
        window->getCd()->setStftZone(zone_);
        window->getPipe()->request_filter2D_roi_update();
        window->getPipe()->request_filter2D_roi_end();
        if (parent_->getCd()->filter_2d_type == Filter2DType::BandPass)
            return;
    }

    parent_->getCd()->fft_shift_enabled = false;

    active_ = false;
}
} // namespace gui
} // namespace holovibes
