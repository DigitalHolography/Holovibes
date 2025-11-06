/*! \file
 *
 * \brief Overlay used to edit the delete twin image rectangle.
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class DeleteTwinMaskOverlay
 *
 * \brief Rectangular overlay to configure delete twin image masks.
 */
class DeleteTwinMaskOverlay : public RectOverlay
{
  public:
    explicit DeleteTwinMaskOverlay(BasicOpenGLWindow* parent);

    void onSetCurrent() override;
    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
