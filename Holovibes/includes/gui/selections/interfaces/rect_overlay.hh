/*! \file
 *
 * \brief Interface for all outlined rectangular overlays.
 *
 * You can control:
 * - The color of the edges with the variable `color_`.
 * - The opacity of the edges with the variable `alpha_`.
 *
 * If you want a filled rectangle look at \see holovibes::gui::FilledRectOverlay
 */
#pragma once

#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class RectOverlay
 *
 * \brief Implementation of an outliner rectangular overlay
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
