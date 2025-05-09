/*! \file
 *
 * \brief Overlay used to display a reticle in the center of the window.
 */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "circ_overlay.hh"

namespace holovibes::gui
{
/*! \class ContrastReticleOverlay
 *
 * \brief Class that represents a reticle overlay in the window.
 */
class ContrastReticleOverlay : public CircOverlay
{
  public:
    ContrastReticleOverlay(BasicOpenGLWindow* parent);
    virtual ~ContrastReticleOverlay() {}

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;
};

/*! \class ReticleOverlay
 *
 * \brief Class that represents a reticle overlay in the window.
 */
class ReticleOverlay : public CircOverlay
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
