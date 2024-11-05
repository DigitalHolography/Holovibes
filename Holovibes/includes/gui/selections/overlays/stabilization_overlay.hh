/*! \file
 *
 * \brief Overlay used to display a circle for the stabilization mask at the center of the window.
 */
#pragma once

#include "circ_overlay.hh"

namespace holovibes::gui
{
/*! \class StabilizationOverlay
 *
 * \brief Overlay used to display a circle for the stabilization mask at the center of the window.
 */
class StabilizationOverlay : public CircOverlay
{
  public:
    StabilizationOverlay(BasicOpenGLWindow* parent);
    virtual ~StabilizationOverlay() {}

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;
};
} // namespace holovibes::gui
