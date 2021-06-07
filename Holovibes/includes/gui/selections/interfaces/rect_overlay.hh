/*! \file
 *
 * Interface for all rectangular overlays. */
#pragma once

#include "Overlay.hh"

namespace holovibes
{
namespace gui
{
class RectOverlay : public Overlay
{
  public:
    RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    virtual ~RectOverlay() {}

    virtual void init() override;
    virtual void draw() override;

    virtual void move(QMouseEvent* e) override;
    virtual void checkCorners();

  protected:
    void setBuffer() override;
};
} // namespace gui
} // namespace holovibes
