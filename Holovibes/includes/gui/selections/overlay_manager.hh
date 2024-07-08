/*! \file
 *
 * \brief Wrapper around a vector of overlays, allowing manipulation of all overlays of a window at once.
 */
#pragma once

#include "Overlay.hh"

namespace holovibes::gui
{
/*! \class OverlayManager
 *
 * \brief Manages a collection of overlays for a window.
 *
 * This class provides functionality to create, manage, and interact with multiple overlays
 * within a window. It supports creating overlays, enabling/disabling them, handling user
 * input events, and drawing overlays.
 */
class OverlayManager
{
  public:
    /*! \brief Constructor
     *
     * \param parent Pointer to the parent BasicOpenGLWindow.
     */
    OverlayManager(BasicOpenGLWindow* parent);

    /*! \brief Destructor */
    ~OverlayManager();

    /*! \brief Creates an overlay based on the specified kind.
     *
     * \tparam ko The kind of overlay to create.
     */
    template <KindOfOverlay ko>
    void create_overlay();

    /*! \brief Creates the default overlay in the view.
     *
     * Zoom for Raw/Holo, Cross for Slices.
     */
    void create_default();

    /*! \brief Disables all overlays of the specified kind.
     *
     * \param ko The kind of overlay to disable.
     * \return True if overlays were disabled, false otherwise.
     */
    bool disable_all(KindOfOverlay ko);

    /*! \brief Enables all overlays of the specified kind.
     *
     * \param ko The kind of overlay to enable.
     * \return True if overlays were enabled, false otherwise.
     */
    bool enable_all(KindOfOverlay ko);

    /*! \brief Disables all overlays and optionally creates a default overlay.
     *
     * \param def If true, creates a default overlay.
     */
    void reset(bool def = true);

    /*! \brief Handles mouse press events for the current overlay.
     *
     * \param e Pointer to the QMouseEvent.
     */
    void press(QMouseEvent* e);

    /*! \brief Handles key press events for the current overlay.
     *
     * \param e Pointer to the QKeyEvent.
     */
    void keyPress(QKeyEvent* e);

    /*! \brief Handles mouse move events for the current overlay.
     *
     * \param e Pointer to the QMouseEvent.
     */
    void move(QMouseEvent* e);

    /*! \brief Handles mouse release events for the current overlay.
     *
     * \param frameSide The frame side where the release occurred.
     */
    void release(ushort frameSide);

    /*! \brief Draws all the overlays that should be displayed. */
    void draw();

    /*! \brief Gets the zone of the current overlay.
     *
     * \return The zone of the current overlay.
     */
    units::RectWindow getZone() const;

    /*! \brief Gets the kind of the current overlay.
     *
     * \return The kind of the current overlay.
     */
    KindOfOverlay getKind() const;

#ifdef _DEBUG
    /*! \brief Prints every overlay in the vector for debugging purposes. */
    void printVector();
#endif

  private:
    /*! \brief Adds a newly created overlay to the vector, sets it as the current overlay, and initializes it.
     *
     * \param new_overlay The newly created overlay.
     */
    void create_overlay(std::shared_ptr<Overlay> new_overlay);

    /*! \brief Sets the current overlay and notifies observers to update the GUI.
     *
     * \param new_overlay The new current overlay.
     */
    void set_current(std::shared_ptr<Overlay> new_overlay);

    /*! \brief Attempts to set the current overlay to the first active overlay of the specified type.
     *
     * \param ko The kind of overlay to set as current.
     * \return True if the current overlay was set, false otherwise.
     */
    bool set_current(KindOfOverlay ko);

    /*! \brief Deletes all disabled overlays from the vector. */
    void clean();

    std::vector<std::shared_ptr<Overlay>> overlays_; /*!< Collection of created overlays. */
    std::shared_ptr<Overlay> current_overlay_; /*!< The current overlay in use. */
    BasicOpenGLWindow* parent_; /*!< Pointer to the parent window. */
};
} // namespace holovibes::gui