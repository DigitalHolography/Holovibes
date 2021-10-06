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
#include "Filter2DWindow.hh"
#include "import_panel.hh"
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

    Ui::MainWindow ui;                                         // RIEN
    Holovibes& holovibes_;                                     // RIEN
    ComputeDescriptor& cd_;                                    // RIEN (TOUT?)
    camera::FrameDescriptor file_fd_;                          // RIEN (I/IR/V?)
    std::unique_ptr<RawWindow> mainDisplay = nullptr;          // RIEN
    std::unique_ptr<SliceWindow> sliceXZ = nullptr;            // RIEN
    std::unique_ptr<SliceWindow> sliceYZ = nullptr;            // RIEN
    std::unique_ptr<RawWindow> lens_window = nullptr;          // RIEN
    std::unique_ptr<RawWindow> raw_window = nullptr;           // RIEN
    std::unique_ptr<Filter2DWindow> filter2d_window = nullptr; // RIEN
    std::unique_ptr<PlotWindow> plot_window_ = nullptr;        // RIEN

    uint window_max_size = 768;                          // RIEN
    uint time_transformation_cuts_window_max_size = 512; // RIEN
    uint auxiliary_window_max_size = 512;                // RIEN

    float displayAngle = 0.f; // V?
    float xzAngle = 0.f;      // V?
    float yzAngle = 0.f;      // V?

    int displayFlip = 0; // V?
    int xzFlip = 0;      // V?
    int yzFlip = 0;      // V?

    bool is_enabled_camera_ = false; // RIEN?IR?
    double z_step_ = 0.005f;         // IR

    bool is_recording_ = false;                // RIEN?
    unsigned record_frame_step_ = 512;         // E?
    RecordMode record_mode_ = RecordMode::RAW; // E

    std::string default_output_filename_; // E
    std::string record_output_directory_; // E
    std::string file_input_directory_;    // E
    std::string batch_input_directory_;   // E

    enum ImportType // I
    {
        None,
        Camera,
        File,
    };
    CameraKind kCamera = CameraKind::NONE;      // RIEN?IR?
    ImportType import_type_ = ImportType::None; // I
    QString last_img_type_ = "Magnitude";       // V

    size_t auto_scale_point_threshold_ = 100; // RIEN
    ushort theme_index_ = 0;                  // RIEN

    // Shortcuts (initialized in constructor)
    QShortcut* z_up_shortcut_;    // ?
    QShortcut* z_down_shortcut_;  // ?
    QShortcut* p_left_shortcut_;  // ?
    QShortcut* p_right_shortcut_; // ?
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
    void configure_holovibes();
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
    void set_image_mode(QString mode); // IR

    void refreshViewMode(); // ?

    /*! \brief Enable the convolution mode
     *
     * \param enable true: enable, false: disable
     */
    void set_convolution_mode(const bool enable); // IR

    /*! \brief Enable the divide convolution mode
     *
     * \param value true: enable, false: disable
     */
    void set_divide_convolution_mode(const bool value); // IR

    /*! \brief Enables or Disables renormalize image with clear image accumulation pipe
     *
     * \param value true: enable, false: disable
     */
    void toggle_renormalize(bool value); // V

    /*! \brief Checks if we are currently in raw mode
     *
     * \return true if we are in raw mode, false otherwise
     */
    bool is_raw_mode(); // RIEN

    /*! \brief Resets the whole program in reload .ini file */
    void reset(); // RIEN

    /*! \brief adds or removes filter 2d view
     *
     * \param checked true: enable, false: disable
     */
    void update_filter2d_view(bool checked); // IR

    /*! \brief Deactivates filter2d view */
    void disable_filter2d_view(); // RIEN (IR?)

    /*! \brief Applies or removes 2d filter on output display
     *
     * \param checked true: enable, false: disable
     */
    void set_filter2d(bool checked); // IR

    /*! \brief Modifies filter2d n1 (first value)
     *
     * \param n The new filter2d n1 value
     */
    void set_filter2d_n1(int n); // IR

    /*! \brief Modifies filter2d n2 (second value)
     *
     * \param n The new filter2d n2 value
     */
    void set_filter2d_n2(int n); // IR

    /*! \brief Removes 2d filter on output display */
    void cancel_filter2d(); // RIEN (IR)

    /*! \brief Changes the time transformation size from ui value */
    void set_time_transformation_size(); // IR

    /*! \brief Adds or removes lens view
     *
     * \param value true: add, false: remove
     */
    void update_lens_view(bool value); // V

    /*! \brief Removes lens view */
    void disable_lens_view(); // RIEN (V?)

    /*! \brief Adds or removes raw view
     *
     * \param value true: add, false: remove
     */
    void update_raw_view(bool value); // V

    /*! \brief Removes raw view */
    void disable_raw_view(); // RIEN (V?)

    /*! \brief Modifies p accumulation from ui value */
    void set_p_accu(); // V

    /*! \brief Modifies x accumulation from ui value */
    void set_x_accu(); // V

    /*! \brief Modifies y accumulation from ui value */
    void set_y_accu(); // V

    /*! \brief Modifies x and y from ui values */
    void set_x_y(); // V

    /*! \brief Modifies q accumulation from ui value
     *
     * \param value The new q value
     */
    void set_q(int value); // V

    /*! \brief Modifies q accumulation from ui value */
    void set_q_acc(); // V

    /*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values */
    void set_composite_intervals(); // C

    /*! \brief Modifies HSV Hue min frequence */
    void set_composite_intervals_hsv_h_min(); // C

    /*! \brief Modifies HSV Hue max frequence*/
    void set_composite_intervals_hsv_h_max(); // C

    /*! \brief Modifies HSV Saturation min frequence */
    void set_composite_intervals_hsv_s_min(); // C

    /*! \brief Modifies HSV Saturation max frequence */
    void set_composite_intervals_hsv_s_max(); // C

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_min(); // C

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_max(); // C

    /*! \brief Modifies the RGV from ui values */
    void set_composite_weights(); // C

    /*! \brief Automatic equalization (Auto-constrast)
     *
     * \param value true: enable, false: disable
     */
    void set_composite_auto_weights(bool value); // C

    /*! \brief Switchs between RGB mode and HSV mode */
    void click_composite_rgb_or_hsv(); // C

    /*! \brief Modifies Hue min threshold and guaratees that Hue min threshold does not exceed Hue max threshold */
    void slide_update_threshold_h_min(); // C

    /*! \brief Modifies Hue max threshold and guaratees that Hue max threshold is higher than Hue min threshold */
    void slide_update_threshold_h_max(); // C

    /*! \brief Change Saturation min threshold. Saturation min threshold does not exceed max threshold */
    void slide_update_threshold_s_min(); // C

    /*! \brief Change Saturation max. Saturation max threshold is higher than min threshold */
    void slide_update_threshold_s_max(); // C

    /*! \brief Change Value min threshold and guarantees that Value min threshold does not exceed Value max threshold */
    void slide_update_threshold_v_min(); // C

    /*! \brief Modifies Value max threshold and guarantees that Value max threshold is higher than Value min threshold
     */
    void slide_update_threshold_v_max(); // C

    /*! \brief Enables or disables Saturation frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_s(); // C

    /*! \brief Enables or disables Value frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_v(); // C

    /*! \brief Enables or disables Hue gaussian blur from ui checkbox */
    void actualize_checkbox_h_gaussian_blur(); // C

    /*! \brief Modified Hue blur size from ui value */
    void actualize_kernel_size_blur(); // C

    /*! \brief Modifies p from ui value
     *
     * \param value The new value of p
     */
    void set_p(int value); // V

    /*! \brief Increment p by 1 on key shortcut */
    void increment_p(); // V

    /*! \brief Decrement p by 1 on key shortcut */
    void decrement_p(); // V

    /*! \brief Modifies wave length (lambda)
     *
     * \param value The new value of lambda
     */
    void set_wavelength(double value); // IR

    /*! \brief Modifies z from ui value
     *
     * \param value The new value of z
     */
    void set_z(double value); // IR

    /*! \brief Increment z by 1 on key shortcut */
    void increment_z(); // IR

    /*! \brief Decrement z by 1 on key shortcut */
    void decrement_z(); // IR

    /*! \brief Modifies the z step on scroll
     *
     * \param value The new incrementation/decrementation step
     */
    void set_z_step(double value); // IR

    /*! \brief Modifies space transform calculation
     *
     * \param value The new space transform to apply
     */
    void set_space_transformation(const QString& value); // IR

    /*! \brief Modifies time transform calculation
     *
     * \param value The new time transform to apply
     */
    void set_time_transformation(const QString& value); // IR

    /*! \brief Enables or Disables time transform cuts views
     *
     * \param checked true: enable, false: disable
     */
    void toggle_time_transformation_cuts(bool checked); // V

    /*! \brief Disables time transform cuts views */
    void cancel_stft_slice_view(); // RIEN (V?)

    /*! \brief Modifies batch size from ui value */
    void update_batch_size(); // IR

    /*! \brief Modifies time transformation stride size from ui value */
    void update_time_transformation_stride(); // IR

    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_mode(QString value); // V

    /*! \brief Enables or Disables unwrapping 2d
     *
     * \param value true: enable, false: disable
     */
    void set_unwrapping_2d(bool value); // V

    /*! \brief Enables or Disables accumulation for the current window
     *
     * \param value true: enable, false: disable
     */
    void set_accumulation(bool value); // V

    /*! \brief Modifies the accumulation level on the current window
     *
     * \param value The new level value
     */
    void set_accumulation_level(int value); // V

    /*! \brief Enables or Disables the contrast mode and update the current focused window
     *
     * \param value true: enable, false: disable
     */
    void set_contrast_mode(bool value); // V

    /*! \brief Enalbles auto-contrast */
    void set_auto_contrast(); // V

    /*! \brief Modifies the min contrast value on the current window
     *
     * \param value The new min contrast value
     */
    void set_contrast_min(double value); // V

    /*! \brief Modifies the max contrast value on the current window
     *
     * \param value the new max contrast value
     */
    void set_contrast_max(double value); // V

    /*! \brief Enables or Disables contrast invertion
     *
     * \param value true: enable, false: disable
     */
    void invert_contrast(bool value); // V

    /*! \brief Enables or Disables auto refresh contrast
     *
     * \param value true: enable, false: disable
     */
    void set_auto_refresh_contrast(bool value); // V

    /*! \brief Enables or Disables log scale on the current window
     *
     * \param value true: enable, false: disable
     */
    void set_log_scale(bool value); // V

    /*! \brief Enables or Disables fft shift mode on the main display window
     *
     * \param value true: enable, false: disable
     */
    void set_fft_shift(bool value); // V

    /*! \brief Make the ui compisite overlay visible */
    void set_composite_area(); // C

    /*! \brief Modifies convolution kernel
     *
     * \param value The new kernel to apply
     */
    void update_convo_kernel(const QString& value); // C

    /*! \brief Modifies the z step on scroll
     *
     * \param value the new incrementation/decrementation step
     */
    void set_record_frame_step(int value); // E

    /*! \brief Changes the focused windows */
    void change_window(); // RIEN

    /*! \brief Browses to import/ export .ini file */
    void browse_import_ini(); // RIEN
    void browse_export_ini(); // RIEN

    /*! \brief Reloads .ini file that store program's state */
    void reload_ini(QString filename); // RIEN
    void reload_ini();                 // RIEN

    /*! \brief Saves the current state of holovibes in .ini file */
    void write_ini(QString filename); // RIEN
    void write_ini();                 // RIEN

    /*! \brief Changes the theme of the ui */
    void set_classic(); // RIEN

    /*! \brief Changes the theme of the ui */
    void set_night(); // RIEN

    /*! \brief Rotates the current selected output display window (XYview or XZview or YZview) */
    void rotateTexture(); // V

    /*! \brief Flips the current selected output display window (XYview or XZview or YZview) */
    void flipTexture(); // V

    /*! \brief Creates or Removes the reticle overlay
     *
     * \param value true: create, false: remove
     */
    void display_reticle(bool value); // V

    /*! \brief Modifies reticle scale in ]0, 1[
     *
     * \param value The new reticle scale
     */
    void reticle_scale(double value); // V

    /*! \brief Opens file explorer on the fly to let the user chose the output file he wants with extension
     * replacement*/
    void browse_record_output_file(); // E

    /*! \brief Enables or Disables number of frame restriction for recording
     *
     * \param value true: enable, false: disable
     */
    void set_nb_frames_mode(bool value); // E

    /*! \brief Modifies the record mode
     *
     * \param value The new record mode
     */
    void set_record_mode(const QString& value); // E

    /*! \brief Stops the record */
    void stop_record(); // E

    /*! \brief Resets ui on record finished
     *
     * \param record_mode The current record mode
     */
    void record_finished(RecordMode record_mode); // RIEN (E?)

    /*! \brief Starts recording */
    void start_record(); // E

    /*! \brief Browses output file */
    void browse_batch_input(); // E

    /*! \brief Creates Signal overlay */
    void activeSignalZone(); // E

    /*! \brief Creates Noise overlay */
    void activeNoiseZone(); // E

    /*! \brief Opens Chart window */
    void start_chart_display(); // E

    /*! \brief Closes Chart window */
    void stop_chart_display(); // E
    /*! \} */

#pragma endregion
    /* ---------- */
  signals:
    /*! \brief TODO: comment
     *
     * \param f
     */
    void synchronize_thread_signal(std::function<void()> f); // RIEN
#pragma region Protected / Private Methods
  protected:
    /*! \brief Last call before the program is closed
     *
     * \param event Unused
     */
    virtual void closeEvent(QCloseEvent* event) override; // RIEN

  public:
    /*! \brief Changes display mode to Raw */
    void set_raw_mode(); // IR

    /*! \brief Changes display mode to Holographic */
    void set_holographic_mode(); // IR

    /*! \brief Set computation mode from ui value (Raw or Holographic) */
    void set_computation_mode(); // IR

    /*! \brief Sets camera frame timout */
    void set_camera_timeout(); // RIEN

    /*! \brief Changes camera
     *
     * \param c The new camera
     */
    void change_camera(CameraKind c); // RIEN

    /*! \brief Opens a file
     *
     * \param path The path of the file to open
     */
    void open_file(const std::string& path); // RIEN

    /*! \brief Setups program from .ini file
     *
     * \param path The path where the .ini file is
     */
    void load_ini(const std::string& path); // RIEN

    /*! \brief Saves the current state of holovibes
     *
     * \param path The location of the .ini file saved
     */
    void save_ini(const std::string& path); // RIEN

    /*! \brief Remove time transformation cut views */
    void cancel_time_transformation_cuts(); // RIEN (V?)

    /*! \brief Creates the pipeline */
    void createPipe(); // RIEN

    /*! \brief Creates the windows for processed image output */
    void createHoloWindow(); // RIEN

    /*! \brief Closes all the displayed windows */
    void close_windows(); // RIEN

    /*! \brief Stops critical compute */
    void close_critical_compute(); // RIEN

    /*! \brief Clears the info container (real time data bench panel) */
    void remove_infos(); // INFO

    /*! \brief Triggers the pipe to make it refresh */
    void pipe_refresh(); // RIEN

    /*! \brief Adds auto contrast to the pipe over cut views */
    void set_auto_contrast_cuts(); // RIEN (V?)

    /*! \brief Enable the filter2d mode */
    void set_filter2d_pipe(); // IR

    /*! \brief Changes Box value without triggering any signal
     *
     * \param spinBox The box to change
     * \param value The value to set
     */
    void QSpinBoxQuietSetValue(QSpinBox* spinBox, int value); // RIEN (MERE?)

    /*! \brief Changes Slider value without triggering any signal
     *
     * \param slider The slider to change
     * \param value The value to set
     */
    void QSliderQuietSetValue(QSlider* slider, int value); // RIEN (MERE?)

    /*! \brief Changes SpinBox value without triggering any signal
     *
     * \param spinBox The spinbox to change
     * \param value The value to set
     */
    void QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value); // RIEN (MERE?)

#pragma endregion
/* ---------- */
#pragma region Fields

#pragma endregion
};
} // namespace gui
} // namespace holovibes
