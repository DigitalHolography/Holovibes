/*! \file
 *
 * \brief Overlay used to display a reticle in the center of the window.
 */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class ReticleOverlay
 *
 * \brief class that represents a reticle overlay in the window.
 */
class ReticleOverlay : public RectOverlay
{
  public:
    ReticleOverlay(BasicOpenGLWindow* parent);
    virtual ~ReticleOverlay() {}

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;
};
} // namespace holovibes::gui
