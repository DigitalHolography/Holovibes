/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
        parent_->getCd()->signalZone(zone_, AccessMode::Set);
}
} // namespace gui
} // namespace holovibes
