/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Overlay ending the band-pass filtering ROI procedure. */
#pragma once

#include "square_overlay.hh"
#include <memory>
#include "filter2d_overlay.hh"

namespace holovibes
{
namespace gui
{
class Filter2DSubZoneOverlay : public RectOverlay
{
  public:
    Filter2DSubZoneOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;

    void setFilter2dOverlay(std::shared_ptr<Filter2DOverlay> rhs);

  private:
    std::shared_ptr<Filter2DOverlay> filter2d_overlay_;
};
} // namespace gui
} // namespace holovibes