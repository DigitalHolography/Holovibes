/*! \file
 *
 * \brief Wrapper around the vector of overlay.
 *
 * Permitting to manipulate all overlays of a window at once.
 */
#pragma once

#include "Overlay.hh"

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

    /*! \brief Create an overlay depending on the value passed to the template. */
    template <KindOfOverlay ko>
    void create_overlay();
    template <>
    void create_overlay<Zoom>();
    template <>
    void create_overlay<Noise>();
    template <>
    void create_overlay<Signal>();
    template <>
    void create_overlay<Cross>();
    template <>
    void create_overlay<SliceCross>();
    template <>
    void create_overlay<KindOfOverlay::CompositeArea>();
    template <>
    void create_overlay<Rainbow>();
    template <>
    void create_overlay<Reticle>();
    template <>
    void create_overlay<Filter2DReticle>();

    /*! \brief Create the default overlay in the view. Zoom for Raw/Holo, Cross for Slices. */
    void create_default();

    /*! \brief Disable all the overlay of kind ko */
    bool disable_all(KindOfOverlay ko);
    /*! \brief Enable all the overlay of kind ko */
    bool enable_all(KindOfOverlay ko);
    /*! \brief Disable all the overlays. If def is set, it will create a default overlay. */
    void reset(bool def = true);

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
    /*! \brief Get the zone of the current overlay. */
    units::RectWindow getZone() const;
    /*! \brief Get the kind of the current overlay. */
    KindOfOverlay getKind() const;

#ifdef _DEBUG
    /*! \brief Prints every overlay in the vector. Debug purpose. */
    void printVector();
#endif

  private:
    /*! \brief Push in the vector the newly created overlay, set the current overlay, and call its init function. */
    void create_overlay(std::shared_ptr<Overlay> new_overlay);

    /*! \brief Set the current overlay and notify observers to update gui. */
    void set_current(std::shared_ptr<Overlay> new_overlay);
    /*! \brief Try to set the current overlay to the first active overlay of a given type. */
    bool set_current(KindOfOverlay ko);

    /*! \brief Deletes from the vector every disabled overlay. */
    void clean();

    /*! \brief Containing every created overlay. */
    std::vector<std::shared_ptr<Overlay>> overlays_;
    /*! \brief Current overlay used by the user. */
    std::shared_ptr<Overlay> current_overlay_;

    /*! \brief Parent window
     *
     * When we delete BasicOpenGlWindow which contains an instance of this,
     * we cannot have a pointer to it otherwise it will never be destroyed.
     * We could use weak_ptr instead of raw pointer.
     */
    BasicOpenGLWindow* parent_;
};
} // namespace holovibes::gui
