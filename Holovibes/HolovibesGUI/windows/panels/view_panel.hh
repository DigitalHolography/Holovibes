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
    void close_windows();

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    /*! \brief Remove time transformation cut views */
    void cancel_time_transformation_cuts();
    /*! \brief Adds auto contrast to the pipe over cut views */
    void set_auto_contrast_cuts();

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

    void update_x_accu();
    void update_y_accu();
    void update_p_accu();
    void update_q_accu();

    void increment_p_index();
    void decrement_p_index();

    /*! \brief Rotates the current selected output display window (ViewXY or ViewXZ or ViewYZ) */
    void rotateTexture();

    /*! \brief Flips the current selected output display window (ViewXY or ViewXZ or ViewYZ) */
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
    void update_accumulation_level();

    /*! \brief Enables or Disables the contrast mode and update the current focused window
     *
     * \param value true: enable, false: disable
     */
    void set_contrast_mode(bool value);

    void request_exec_contrast_current_window();
    void update_contrast_current_windows_range();

    /*! \brief Enables or Disables auto refresh contrast
     *
     * \param value true: enable, false: disable
     */
    void set_auto_refresh_contrast(bool value);
    /*! \brief Enables or Disables contrast invertion
     *
     * \param value true: enable, false: disable
     */
    void invert_contrast(bool value);

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
    void update_reticle_scale();

  private:
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;
};
} // namespace holovibes::gui
