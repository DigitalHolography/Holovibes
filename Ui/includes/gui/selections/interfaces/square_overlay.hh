/*! \file
 *
 * \brief Interface for all square overlays.
 */
#pragma once

#include "filled_rect_overlay.hh"

namespace holovibes::gui
{
/*! \class SquareOverlay
 *
 * \brief Class that represents a square overlay in the window.
 */
class SquareOverlay : public FilledRectOverlay
{
  public:
    SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    virtual ~SquareOverlay() {}

    /*! \brief Check if corners don't go out of bounds. */
    void checkCorners() override;
    /*! \brief Change the rectangular zone to a square zone, using the shortest side */
    void make_square();

    virtual void move(QMouseEvent* e) override;
};
} // namespace holovibes::gui
