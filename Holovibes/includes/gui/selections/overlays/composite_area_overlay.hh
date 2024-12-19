/*! \file
 *
 * \brief Overlay selecting the zone to normalize colors.
 */
#pragma once

#include "square_overlay.hh"

namespace holovibes::gui
{
/*! \class CompositeAreaOverlay
 *
 * \brief class that represents a composite area overlay in the window.
 */
class CompositeAreaOverlay : public RectOverlay
{
  public:
    CompositeAreaOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
