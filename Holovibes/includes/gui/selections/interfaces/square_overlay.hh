/*! \file
 *
 * \brief Interface for all square overlays in the Holovibes GUI.
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class SquareOverlay
 *
 * \brief Manages square-shaped overlays.
 *
 * This class provides functionality specific to overlays that maintain a square shape,
 * extending the capabilities of RectOverlay.
 */
class SquareOverlay : public RectOverlay
{
  public:
    /*! \brief Constructor
     *
     * \param overlay The kind of overlay.
     * \param parent Pointer to the parent BasicOpenGLWindow.
     */
    SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    /*! \brief Destructor */
    virtual ~SquareOverlay() {}

    /*! \brief Checks if the corners of the square are within bounds. */
    void checkCorners() override;

    /*! \brief Converts the rectangular zone to a square zone, using the shortest side. */
    void make_square();

    /*! \brief Handles mouse move events.
     *
     * \param e Pointer to the QMouseEvent.
     */
    virtual void move(QMouseEvent* e) override;
};
} // namespace holovibes::gui