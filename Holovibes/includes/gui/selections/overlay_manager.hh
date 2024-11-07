/*! \file
 *
 * \brief Wrapper around the vector of overlay.
 *
 * Permitting to manipulate all overlays of a window at once.
 */
#pragma once

#include "Overlay.hh"

#include <QTimer>

namespace holovibes::gui
{
/*! \class OverlayManager
 *
 * \brief #TODO Add a description for this class
 */
class OverlayManager
{
  public:
    OverlayManager(BasicOpenGLWindow* parent);
    ~OverlayManager();

    /*! \brief Create an overlay if it does not exist already, activate it and make it visible.
     *
     * \param[in] make_current If true, the overlay will be set as the current overlay.
     * \param[in] ms The time in ms before the overlay will disappear.
     */
    template <KindOfOverlay ko>
    void enable(bool make_current = true, int ms = -1);

    /*! \brief Disable all the overlay of kind ko */
    bool disable(KindOfOverlay ko);

    /*! \brief Create the default overlay in the view if it doesn't exist. Zoom for Raw/Holo, Cross for Slices. */
    void create_default();

    /*! \brief Call the press function of the current overlay. */
    void press(QMouseEvent* e);
    /*! \brief Call the keyPress function of the current overlay. */
    void keyPress(QKeyEvent* e);
    /*! \brief Call the move function of the current overlay. */
    void move(QMouseEvent* e);
    /*! \brief Call the release function of the current overlay. */
    void release(ushort frameSide);

    /*! \brief Draw every overlay that should be displayed. */
    void draw();

    /*! \brief Get the kind of the current overlay. */
    KindOfOverlay getKind() const;

#ifdef _DEBUG
    /*! \brief Prints every overlay in the vector. Debug purpose. */
    void printVector();
#endif

  private:
    /*! \brief Create an overlay of kind ko and return it.
     *
     * \param[in] ko The kind of overlay to create.
     * \return The created overlay.
     */
    std::shared_ptr<Overlay> create_overlay(KindOfOverlay ko);

    /*! \brief Hide overlays when the timer timeout. */
    void hide_overlay();

    /*! \brief Deletes from the vector every disabled overlay. */
    void clean();

    /*! \brief Containing every created overlay. */
    std::vector<std::shared_ptr<Overlay>> overlays_;

    /*! \brief Current overlay used by the user. */
    std::shared_ptr<Overlay> current_overlay_;

    /*! \brief Timer used to hide Overlay after a delay. */
    QTimer timer_;

    /*! \brief Parent window
     *
     * When we delete BasicOpenGlWindow which contains an instance of this,
     * we cannot have a pointer to it otherwise it will never be destroyed.
     * We could use weak_ptr instead of raw pointer.
     */
    BasicOpenGLWindow* parent_;
};
} // namespace holovibes::gui

#include "overlay_manager.hxx"
