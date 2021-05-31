/*! \file
 *
 * Overlay selecting the zone to zoom in. */
#pragma once

#include "square_overlay.hh"

namespace holovibes
{
namespace gui
{
class ZoomOverlay : public SquareOverlay
{
  public:
    ZoomOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace gui
} // namespace holovibes