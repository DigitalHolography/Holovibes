/*! \file
 *
 * Overlay selecting the noise zone to chart. */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
namespace gui
{
class NoiseOverlay : public RectOverlay
{
  public:
    NoiseOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace gui
} // namespace holovibes
