/*! \file
 *
 * Interface for all square overlays. */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
namespace gui
{
class SquareOverlay : public RectOverlay
{
  public:
    SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    virtual ~SquareOverlay() {}

    /*! \brief Check if corners don't go out of bounds. */
    void checkCorners() override;
    /*! \brief Change the rectangular zone to a square zone, using the shortest
     * side */
    void make_square();

    virtual void move(QMouseEvent* e) override;
};
} // namespace gui
} // namespace holovibes
