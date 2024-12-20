/*! \file
 *
 * \brief Overlay selecting the zone to zoom in.
 */
#pragma once

#include "square_overlay.hh"

namespace holovibes::gui
{
/*! \class ZoomOverlay
 *
 * \brief Class that represents a zoom overlay in the window.
 */
class ZoomOverlay : public SquareOverlay
{
  public:
    ZoomOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
