/*! \file
 *
 * \brief Interface for all rectangular overlays in the Holovibes GUI.
 */
#pragma once

#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class RectOverlay
 *
 * \brief Manages rectangular-shaped overlays.
 *
 * This class provides functionality for overlays that maintain a rectangular shape,
 * extending the capabilities of the base Overlay class.
 */
class RectOverlay : public Overlay
{
  public:
    /*! \brief Constructor
     *
     * \param overlay The kind of overlay.
     * \param parent Pointer to the parent BasicOpenGLWindow.
     */
    RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    /*! \brief Destructor */
    virtual ~RectOverlay() {}

    /*! \brief Initializes the rectangular overlay. */
    virtual void init() override;

    /*! \brief Draws the rectangular overlay using OpenGL functions. */
    virtual void draw() override;

    /*! \brief Handles mouse move events.
     *
     * \param e Pointer to the QMouseEvent.
     */
    virtual void move(QMouseEvent* e) override;

    /*! \brief Checks if the corners of the rectangle are within bounds. */
    virtual void checkCorners();

  protected:
    /*! \brief Sets the vertex buffer with the coordinates of the rectangle. */
    void setBuffer() override;
};
} // namespace holovibes::gui