/*! \file
 *
 * \brief Overlay used to compute the side views.*
 */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"
#include "zoom_overlay.hh"

namespace holovibes
{
namespace gui
{
/*! \class CrossOverlay
 *
 * \brief #TODO Add a description for this class
 */
class CrossOverlay : public Overlay
{
  public:
    CrossOverlay(BasicOpenGLWindow* parent);
    virtual ~CrossOverlay();

    /*! \brief Initialize opengl buffers for rectangles and lines.
     *  The vertices buffers is built like this:
     *
     *       0   1
     *       |   |
     *  4 --- --- --- 5
     *       |   |
     *  7 --- --- --- 6
     *       |   |
     *       3   2
     */
    void init() override;
    void draw() override;

    void onSetCurrent() override;

    void press(QMouseEvent* e) override;
    void keyPress(QKeyEvent* e) override;
    void move(QMouseEvent* e) override;
    void release(ushort frameSide) override;

  protected:
    void setBuffer() override;

    /*! \brief Computes the zones depending on compute descriptor of the parent
     */
    void computeZone();

    //! Transparency of the borders
    float line_alpha_;

    //! Vertices order for lines
    GLuint elemLineIndex_;

    //! Locking line overlay
    bool locked_;

    //! Actual mouse position
    units::PointFd mouse_position_{0, 0};

    //! Horizontal area. zone_ corresponds to the vertical area
    units::RectFd horizontal_zone_;
};
} // namespace gui
} // namespace holovibes
