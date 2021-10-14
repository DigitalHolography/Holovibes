/*! \file
 *
 * \brief Qt main class containing the GUI.
 */
#pragma once

// without namespace
#include "tools.hh"
#include "json.hh"
using json = ::nlohmann::json;

#include "enum_record_mode.hh"

// namespace camera
#include "camera_exception.hh"

// namespace holovibes
#include "holovibes.hh"
#include "custom_exception.hh"

// namespace gui
#include "HoloWindow.hh"
#include "SliceWindow.hh"
#include "PlotWindow.hh"
#include "AdvancedSettingsWindow.hh"
#include "Filter2DWindow.hh"
#include "ui_mainwindow.h"

Q_DECLARE_METATYPE(std::function<void()>)

namespace holovibes
{
namespace gui
{
/*! \class MainWindow
 *
 * \brief Main class of the GUI. It regroup most of the Qt slots used for user actions.
 *
 * These slots are divided into several sections:
 *
 * * Menu: every action in the menu (e-g: configuration of .ini, camera selection ...).
 * * Image rendering: #img, p, z, lambda ...
 * * View: log scale, shifted corner, contrast ...
 * * Special: image ratio, Chart plot ...
 * * Record: record of raw frames, Chart file ...
 * * Import : making a file of raw data the image source
 * * Info : Various runtime informations on the program's state
 */
class MainWindow : public QMainWindow, public Observer
{
    Q_OBJECT
/* ---------- */
#pragma region Public Methods
  public:
    /*! \brief Set keyboard shortcuts, set visibility and load default values from holovibes.ini.
     *
     * \param holovibes holovibes object
     * \param parent Qt parent (should be null as it is the GUI hierarchy top class)
     */
    MainWindow(Holovibes& holovibes, QWidget* parent = 0);
    ~MainWindow();

    void notify() override;
    void notify_error(const std::exception& e) override;

    RawWindow* get_main_display();

    friend class AdvancedSettingsWindow;
#pragma endregion
/* ---------- */
#pragma region Public Slots
  public slots:
    void on_notify();
    void update_file_reader_index(int n);
    /*! \brief Give a function to execute to the main thread with a signal
     *
     * \param f the function to execute
     */
    void synchronize_thread(std::function<void()> f);
    /*! \brief Resize windows if one layout is toggled. */
    void layout_toggled();
    void camera_none();
    void camera_adimec();
    void camera_ids();
    void camera_phantom();
    void camera_bitflow_cyton();
    void camera_hamamatsu();
    void camera_xiq();
    void camera_xib();
    void configure_camera();

    /*! \brief Opens the credit display */
    void credits();
    void documentation();
    void init_image_mode(QPoint& position, QSize& size);

    /*! \name Image Rendering
     * \{
     *
     * Lots of these methods stick to the following scheme:
     *
     * * Get pipe
     * * Set visibility to false
     * * Check if value is correct/into slot specific bounds
     * * Update a value in FrameDescriptor of the holovibes object
     * * Request a pipe refresh
     * * Set visibility to true
     */

    /*! \brief Set image mode either to raw or hologram mode
     *
     * Check if Camera has been enabled, then create a new GuiGLWindow keeping
     * its old position and size if it was previously opened, set visibility
     * and call notify().
     *
     * \param value true for raw mode, false for hologram mode.
     */
    void set_image_mode(QString mode);

    void refreshViewMode();

    /*! \brief Enable the convolution mode
     *
     * \param enable true: enable, false: disable
     */
    void set_convolution_mode(const bool enable);

    /*! \brief Enable the divide convolution mode
     *
     * \param value true: enable, false: disable
     */
    void set_divide_convolution_mode(const bool value);

    /*! \brief Enables or Disables renormalize image with clear image accumulation pipe
     *
     * \param value true: enable, false: disable
     */
    void toggle_renormalize(bool value);

    /*! \brief Checks if we are currently in raw mode
     *
     * \return true if we are in raw mode, false otherwise
     */
    bool is_raw_mode();

    /*! \brief adds or removes filter 2d view
     *
     * \param checked true: enable, false: disable
     */
    void update_filter2d_view(bool checked);

    /*! \brief Deactivates filter2d view */
    void disable_filter2d_view();

    /*! \brief Applies or removes 2d filter on output display
     *
     * \param checked true: enable, false: disable
     */
    void set_filter2d(bool checked);

    /*! \brief Modifies filter2d n1 (first value)
     *
     * \param n The new filter2d n1 value
     */
    void set_filter2d_n1(int n);

    /*! \brief Modifies filter2d n2 (second value)
     *
     * \param n The new filter2d n2 value
     */
    void set_filter2d_n2(int n);

    /*! \brief Removes 2d filter on output display */
    void cancel_filter2d();

    /*! \brief Changes the time transformation size from ui value */
    void set_time_transformation_size();

    /*! \brief Adds or removes lens view
     *
     * \param value true: add, false: remove
     */
    void update_lens_view(bool value);

    /*! \brief Removes lens view */
    void disable_lens_view();

    /*! \brief Adds or removes raw view
     *
     * \param value true: add, false: remove
     */
    void update_raw_view(bool value);

    /*! \brief Removes raw view */
    void disable_raw_view();

    /*! \brief Modifies p accumulation from ui value */
    void set_p_accu();

    /*! \brief Modifies x accumulation from ui value */
    void set_x_accu();

    /*! \brief Modifies y accumulation from ui value */
    void set_y_accu();

    /*! \brief Modifies x and y from ui values */
    void set_x_y();

    /*! \brief Modifies q accumulation from ui value
     *
     * \param value The new q value
     */
    void set_q(int value);

    /*! \brief Modifies q accumulation from ui value */
    void set_q_acc();

    /*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values */
    void set_composite_intervals();

    /*! \brief Modifies HSV Hue min frequence */
    void set_composite_intervals_hsv_h_min();

    /*! \brief Modifies HSV Hue max frequence*/
    void set_composite_intervals_hsv_h_max();

    /*! \brief Modifies HSV Saturation min frequence */
    void set_composite_intervals_hsv_s_min();

    /*! \brief Modifies HSV Saturation max frequence */
    void set_composite_intervals_hsv_s_max();

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_min();

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_max();

    /*! \brief Modifies the RGV from ui values */
    void set_composite_weights();

    /*! \brief Automatic equalization (Auto-constrast)
     *
     * \param value true: enable, false: disable
     */
    void set_composite_auto_weights(bool value);

    /*! \brief Switchs between RGB mode and HSV mode */
    void click_composite_rgb_or_hsv();

    /*! \brief Modifies Hue min threshold and guaratees that Hue min threshold does not exceed Hue max threshold */
    void slide_update_threshold_h_min();

    /*! \brief Modifies Hue max threshold and guaratees that Hue max threshold is higher than Hue min threshold */
    void slide_update_threshold_h_max();

    /*! \brief Change Saturation min threshold. Saturation min threshold does not exceed max threshold */
    void slide_update_threshold_s_min();

    /*! \brief Change Saturation max. Saturation max threshold is higher than min threshold */
    void slide_update_threshold_s_max();

    /*! \brief Change Value min threshold and guaratees that Value min threshold does not exceed Value max threshold */
    void slide_update_threshold_v_min();

    /*! \brief Modifies Value max threshold and guaratees that Value max threshold is higher than Value min threshold */
    void slide_update_threshold_v_max();

    /*! \brief Enables or disables Saturation frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_s();

    /*! \brief Enables or disables Value frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_v();

    /*! \brief Enables or disables Hue gaussian blur from ui checkbox */
    void actualize_checkbox_h_gaussian_blur();

    /*! \brief Modified Hue blur size from ui value */
    void actualize_kernel_size_blur();

    /*! \brief Modifies p from ui value
     *
     * \param value The new value of p
     */
    void set_p(int value);

    /*! \brief Increment p by 1 on key shortcut */
    void increment_p();

    /*! \brief Decrement p by 1 on key shortcut */
    void decrement_p();

    /*! \brief Modifies wave length (lambda)
     *
     * \param value The new value of lambda
     */
    void set_wavelength(double value);

    /*! \brief Modifies z from ui value
     *
     * \param value The new value of z
     */
    void set_z(double value);

    /*! \brief Increment z by 1 on key shortcut */
    void increment_z();

    /*! \brief Decrement z by 1 on key shortcut */
    void decrement_z();

    /*! \brief Modifies the z step on scroll
     *
     * \param value The new incrementation/decrementation step
     */
    void set_z_step(double value);

    /*! \brief Modifies space transform calculation
     *
     * \param value The new space transform to apply
     */
    void set_space_transformation(const QString& value);

    /*! \brief Modifies time transform calculation
     *
     * \param value The new time transform to apply
     */
    void set_time_transformation(const QString& value);

    /*! \brief Enables or Disables time transform cuts views
     *
     * \param checked true: enable, false: disable
     */
    void toggle_time_transformation_cuts(bool checked);

    /*! \brief Disables time transform cuts views */
    void cancel_stft_slice_view();

    /*! \brief Modifies batch size from ui value */
    void update_batch_size();

    /*! \brief Modifies time transformation stride size from ui value */
    void update_time_transformation_stride();

    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_mode(QString value);

    /*! \brief Enables or Disables unwrapping 2d
     *
     * \param value true: enable, false: disable
     */
    void set_unwrapping_2d(bool value);

    /*! \brief Enables or Disables accumulation for the current window
     *
     * \param value true: enable, false: disable
     */
    void set_accumulation(bool value);

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

    /*! \brief Enalbles auto-contrast */
    void set_auto_contrast();

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

    /*! \brief Enables or Disables contrast invertion
     *
     * \param value true: enable, false: disable
     */
    void invert_contrast(bool value);

    /*! \brief Enables or Disables auto refresh contrast
     *
     * \param value true: enable, false: disable
     */
    void set_auto_refresh_contrast(bool value);

    /*! \brief Enables or Disables log scale on the current window
     *
     * \param value true: enable, false: disable
     */
    void set_log_scale(bool value);

    /*! \brief Enables or Disables fft shift mode on the main display window
     *
     * \param value true: enable, false: disable
     */
    void set_fft_shift(bool value);

    /*! \brief Make the ui compisite overlay visible */
    void set_composite_area();

    /*! \brief Modifies convolution kernel
     *
     * \param value The new kernel to apply
     */
    void update_convo_kernel(const QString& value);

    /*! \brief Modifies the z step on scroll
     *
     * \param value the new incrementation/decrementation step
     */
    void set_record_frame_step(int value);

    /*! \brief Sets the start stop buttons object accessibility
     *
     * \param value accessibility
     */
    void set_start_stop_buttons(bool value);

    /*! \brief Opens file explorer to let the user chose the file he wants to import */
    void import_browse_file();

    /*! \brief Creates an input file to gather data from it.
     *
     * \param filename The chosen file
     */
    void import_file(const QString& filename);

    /*! \brief Sets ui values and constraints + launch FileReadWroker */
    void init_holovibes_import_mode();

    /*! \brief Setups attributes for launching and launchs the imported file */
    void import_start();
    /*! \brief Reset ui and stop holovibes' compute worker and file read worker */
    void import_stop();

    /*! \brief Handles the ui input fps */
    void import_start_spinbox_update();

    /*! \brief Handles the ui output fps */
    void import_end_spinbox_update();

    /*! \brief Changes the focused windows */
    void change_window();

    /*! \brief Browses to import/ export .ini file */
    void browse_import_ini();
    void browse_export_ini();

    /*! \brief Reloads .ini file that store program's state */
    void reload_ini(QString filename);
    void reload_ini();

    /*! \brief Saves the current state of holovibes in .ini file */
    void write_ini(QString filename);
    void write_ini();

    /*! \brief Open/Close Advanced Settings window */
    void open_advanced_settings();
    void close_advanced_settings();

    /*! \brief Changes the theme of the ui */
    void set_classic();

    /*! \brief Changes the theme of the ui */
    void set_night();

    /*! \brief Changes the theme of the ui according to its index */
    void set_theme(const int index);

    /*! \brief Rotates the current selected output display window (XYview or XZview or YZview) */
    void rotateTexture();

    /*! \brief Flips the current selected output display window (XYview or XZview or YZview) */
    void flipTexture();

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

    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    void browse_record_output_file();

    /*! \brief Enables or Disables number of frame restriction for recording
     *
     * \param value true: enable, false: disable
     */
    void set_nb_frames_mode(bool value);

    /*! \brief Modifies the record mode
     *
     * \param value The new record mode
     */
    void set_record_mode(const QString& value);

    /*! \brief Stops the record */
    void stop_record();

    /*! \brief Resets ui on record finished
     *
     * \param record_mode The current record mode
     */
    void record_finished(RecordMode record_mode);

    /*! \brief Starts recording */
    void start_record();

    /*! \brief Browses output file */
    void browse_batch_input();

    /*! \brief Creates Signal overlay */
    void activeSignalZone();

    /*! \brief Creates Noise overlay */
    void activeNoiseZone();

    /*! \brief Opens Chart window */
    void start_chart_display();

    /*! \brief Closes Chart window */
    void stop_chart_display();
    /*! \} */

#pragma endregion
    /* ---------- */
  signals:
    /*! \brief TODO: comment
     *
     * \param f
     */
    void synchronize_thread_signal(std::function<void()> f);
#pragma region Protected / Private Methods
  protected:
    /*! \brief Last call before the program is closed
     *
     * \param event Unused
     */
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    /*! \brief Changes display mode to Raw */
    void set_raw_mode();

    /*! \brief Changes display mode to Holographic */
    void set_holographic_mode();

    /*! \brief Set computation mode from ui value (Raw or Holographic) */
    void set_computation_mode();

    /*! \brief Changes camera
     *
     * \param c The new camera
     */
    void change_camera(CameraKind c);

    /*! \brief Opens a file
     *
     * \param path The path of the file to open
     */
    void open_file(const std::string& path);

    /*! \brief Setups gui from .ini file */
    void load_gui();

    /*! \brief Saves the current state of holovibes */
    void save_gui();

    /*! \brief Remove time transformation cut views */
    void cancel_time_transformation_cuts();

    /*! \brief Creates the pipeline */
    void createPipe();

    /*! \brief Creates the windows for processed image output */
    void createHoloWindow();

    /*! \brief Closes all the displayed windows */
    void close_windows();

    /*! \brief Stops critical compute */
    void close_critical_compute();

    /*! \brief Clears the info container (real time data bench panel) */
    void remove_infos();

    /*! \brief Triggers the pipe to make it refresh */
    void pipe_refresh();

    /*! \brief Adds auto contrast to the pipe over cut views */
    void set_auto_contrast_cuts();

    /*! \brief Enable the filter2d mode */
    void set_filter2d();

    /*! \brief Changes Box value without triggering any signal
     *
     * \param spinBox The box to change
     * \param value The value to set
     */
    void QSpinBoxQuietSetValue(QSpinBox* spinBox, int value);

    /*! \brief Changes Slider value without triggering any signal
     *
     * \param slider The slider to change
     * \param value The value to set
     */
    void QSliderQuietSetValue(QSlider* slider, int value);

    /*! \brief Changes SpinBox value without triggering any signal
     *
     * \param spinBox The spinbox to change
     * \param value The value to set
     */
    void QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value);

#pragma endregion
/* ---------- */
#pragma region Fields

    enum ImportType
    {
        None,
        Camera,
        File,
    };

    Ui::MainWindow ui;
    Holovibes& holovibes_;
    ComputeDescriptor& cd_;
    camera::FrameDescriptor file_fd_;

    std::unique_ptr<RawWindow> mainDisplay = nullptr;
    std::unique_ptr<SliceWindow> sliceXZ = nullptr;
    std::unique_ptr<SliceWindow> sliceYZ = nullptr;
    std::unique_ptr<RawWindow> lens_window = nullptr;
    std::unique_ptr<RawWindow> raw_window = nullptr;
    std::unique_ptr<Filter2DWindow> filter2d_window = nullptr;
    std::unique_ptr<PlotWindow> plot_window_ = nullptr;
    std::unique_ptr<AdvancedSettingsWindow> advanced_settings_window_ = nullptr;

    uint window_max_size = 768;
    uint time_transformation_cuts_window_max_size = 512;
    uint auxiliary_window_max_size = 512;

    bool is_enabled_camera_ = false;
    double z_step_ = 0.005f;

    bool is_recording_ = false;
    unsigned record_frame_step_ = 512;
    RecordMode record_mode_ = RecordMode::RAW;

    std::string default_output_filename_ = "capture";
    std::string record_output_directory_;
    std::string file_input_directory_ = "C:\\";
    std::string batch_input_directory_ = "C:\\";

    CameraKind kCamera = CameraKind::NONE;
    ImportType import_type_ = ImportType::None;
    QString last_img_type_ = "Magnitude";

    size_t auto_scale_point_threshold_ = 100;
    ushort theme_index_ = 0;

    bool is_advanced_settings_displayed = false;
    bool need_close = false;

    // Shortcuts (initialized in constructor)
    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;

    // TODO: Remove the two QSpinBox
    QSpinBox* start_spinbox;
    QSpinBox* end_spinbox;

#pragma endregion
};
} // namespace gui
} // namespace holovibes
