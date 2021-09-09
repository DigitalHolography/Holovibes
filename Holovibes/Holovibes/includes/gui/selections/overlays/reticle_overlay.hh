/*! \file
 *
 * Overlay used to display a reticle in the center of the window. */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"

namespace holovibes
{
namespace gui
{
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

    //! Transparency of the lines
    float alpha_;
};
} // namespace gui
} // namespace holovibes
