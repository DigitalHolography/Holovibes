/*! \file
 *
 * \brief Interface for all rectangular overlays.
 */
#pragma once

#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class RectOverlay
 *
 * \brief #TODO Add a description for this class
 */
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
} // namespace holovibes::gui
