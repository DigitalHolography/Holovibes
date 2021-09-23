/*! \file
 *
 * \brief Overlay manipulating z computation for side views.
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes
{
namespace gui
{
/*! \class SliceCrossOverlay
 *
 * \brief #TODO Add a description for this class
 */
class SliceCrossOverlay : public RectOverlay
{
  public:
    SliceCrossOverlay(BasicOpenGLWindow* parent);
    virtual ~SliceCrossOverlay();

    void init() override;
    void draw() override;

    void keyPress(QKeyEvent* e) override;
    void move(QMouseEvent* e) override;
    void release(ushort frameSide) override;

    void setBuffer() override;

  private:
    /*! \brief Transparency of the borders */
    float line_alpha_;

    /*! \brief Vertices order for lines */
    GLuint elemLineIndex_;

    /*! \brief Locking line overlay */
    bool locked_;

    /*! \brief pindex of the mouse position */
    units::PointFd pIndex_;
};
} // namespace gui
} // namespace holovibes