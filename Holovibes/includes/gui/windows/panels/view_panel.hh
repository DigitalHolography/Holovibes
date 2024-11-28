/*! \file
 *
 * \brief File containing methods, attributes and slots related to the View panel
 */
#pragma once

#include "panel.hh"

#include "HoloWindow.hh"
#include "SliceWindow.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class ViewPanel
 *
 * \brief Class representing the View Panel in the GUI
 */
class ViewPanel : public Panel
{
    Q_OBJECT

  public:
    uint time_transformation_cuts_window_max_size = 512;

    ViewPanel(QWidget* parent = nullptr);
    ~ViewPanel();

    void view_callback(WindowKind kind, ViewWindow window);
    void on_notify() override;

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    /*! \brief Remove time transformation cut views */
    void cancel_time_transformation_cuts();

    /**
     * \brief Changes the image type in the UI and hides irrelevant options.
     * Notably, when the input data is moments, only moments views are available.
     *
     * \param img_type The image type to set.
     */
    void update_img_type(int img_type);

  public slots:
    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_mode(const QString& value);

    /*! \brief Enables or Disables unwrapping 2d
     *
     * \param value true: enable, false: disable
     */
    void set_unwrapping_2d(bool value);
    /*! \brief Enables or Disables time transform cuts views
     *
     * \param checked true: enable, false: disable
     */
    void update_3d_cuts_view(bool checked);
    /*! \brief Enables or Disables fft shift mode on the main display window
     *
     * \param value true: enable, false: disable
     */
    void set_fft_shift(bool value);
    /*!
     * \brief Enables or Disables Artery mask
     *
     */
    void set_artery_mask(bool value);
    /*!
     * \brief Enables or Disables Otsu
     *
     */
    void set_otsu(bool value);
    /*! \brief Enables or Disables registration mode on the main display window.
     *
     * \param value true: enable, false: disable
     */
    void set_registration(bool value);
    /*! \brief Adds or removes lens view
     *
     * \param value true: add, false: remove
     */
    void update_lens_view(bool value);
    /*! \brief Adds or removes raw view
     *
     * \param value true: add, false: remove
     */
    void update_raw_view(bool value);

    /*! \brief Modifies x and y from ui values */
    void set_x_y();
    /*! \brief Modifies x accumulation from ui value */
    void set_x_accu();
    /*! \brief Modifies y accumulation from ui value */
    void set_y_accu();

    /*! \brief Modifies p from ui value */
    void set_p(int value);
    /*! \brief Increment p by 1 on key shortcut */
    void increment_p();
    /*! \brief Decrement p by 1 on key shortcut */
    void decrement_p();
    /*! \brief Modifies p accumulation from ui value */
    void set_p_accu();

    /*! \brief Modifies q accumulation from ui value */
    void set_q(int value);
    /*! \brief Modifies q accumulation from ui value */
    void set_q_acc();

    /*! \brief Rotates the current selected output display window (XYview or XZview or YZview) */
    void rotateTexture();

    /*! \brief Flips the current selected output display window (XYview or XZview or YZview) */
    void flipTexture();

    /*! \brief Enables or Disables log scale on the current window
     *
     * \param value true: enable, false: disable
     */
    void set_log_scale(bool value);

    /*! \brief Modifies the accumulation level on the current window
     *
     * \param value The new level value
     */
    void set_accumulation_level(int value);

    /*! \brief Enables or Disables the contrast mode and update the current focused window
     *
     * \param value true: enable, false: disable
     */
    void set_contrast_mode(bool value);

    /*! \brief Enables or Disables auto refresh contrast
     *
     * \param value true: enable, false: disable
     */
    void set_contrast_auto_refresh(bool value);
    /*! \brief Enables or Disables contrast invertion
     *
     * \param value true: enable, false: disable
     */
    void set_contrast_invert(bool value);

    /*! \brief Modifies the min contrast value on the current window
     *
     * \param value The new min contrast value
     */
    void set_contrast_min(double value);

    /*! \brief Modifies the max contrast value on the current window
     *
     * \param value the new max contrast value
     */
    void set_contrast_max(double value);

    /*! \brief Enables or Disables renormalize image with clear image accumulation pipe
     *
     * \param value true: enable, false: disable
     */
    void toggle_renormalize(bool value);

    /*! \brief Creates or Removes the reticle overlay
     *
     * \param value true: create, false: remove
     */
    void display_reticle(bool value);
    /*! \brief Modifies reticle scale in ]0, 1[
     *
     * \param value The new reticle scale
     */
    void reticle_scale(double value);

    /*! \brief Set the new value of the registration zone for the circular mask. Range ]0, 1[.
     *  \param[in] value The new zone value.
     */
    void update_registration_zone(double value);

  private:
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;
};
} // namespace holovibes::gui
