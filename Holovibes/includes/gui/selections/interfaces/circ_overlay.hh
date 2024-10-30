/*! \file
 *
 * \brief Interface for all outlined circular overlays.
 *
 * The position of the circule is done with the `zone_` variable. `zone_.x()` and `zone_.y()` tells the position of the
 * center and `radius_` the radius.
 *
 * You can control:
 * - The color of the edges with the variable `color_`.
 * - The opacity of the edges with the variable `alpha_`.
 */
#pragma once

#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class RectOverlay
 *
 * \brief Implementation of an outlined circular overlay
 */
class CircOverlay : public Overlay
{
  public:
    CircOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent, uint resolution);

    virtual ~CircOverlay() {}

    virtual void init() override;
    virtual void draw() override;

    void checkBounds();

  protected:
    void setBuffer() override;

    /*! \brief The number of vertices */
    uint resolution_;

    /*! \brief The radius of the circle */
    float radius_;
};
} // namespace holovibes::gui
