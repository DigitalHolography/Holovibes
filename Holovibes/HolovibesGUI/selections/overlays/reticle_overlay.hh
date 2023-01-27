/*! \file
 *
 * \brief Overlay used to display a reticle in the center of the window.
 */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class ReticleOverlay
 *
 * \brief #TODO Add a description for this class
 */
class ReticleOverlay : public Overlay
{
  public:
    ReticleOverlay(BasicOpenGLWindow* parent);
    virtual ~ReticleOverlay() {}

    void init() override;
    void draw() override;

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;

    /*! \brief Transparency of the lines */
    float alpha_;
};
} // namespace holovibes::gui
