/*! \file
 *
 * \brief Overlay manipulating z computation for side views.
 */
#pragma once

#include "filled_rect_overlay.hh"

namespace holovibes::gui
{
/*! \class SliceCrossOverlay
 *
 * \brief class that represents a slice cross overlay in the window.
 */
class SliceCrossOverlay : public FilledRectOverlay
{
  public:
    SliceCrossOverlay(BasicOpenGLWindow* parent);

    void keyPress(QKeyEvent* e) override;
    void move(QMouseEvent* e) override;
    void release(ushort frameSide) override;

    void setBuffer() override;

  private:
    /*! \brief Locking line overlay */
    bool locked_;

    /*! \brief p_index of the mouse position */
    units::PointFd pIndex_;
};
} // namespace holovibes::gui
