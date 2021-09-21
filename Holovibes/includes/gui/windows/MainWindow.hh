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
    void notify_error(std::exception& e) override;

    RawWindow* get_main_display();
#pragma endregion
/* ---------- */
#pragma region Public Slots
  public slots:
    void on_notify();
    void update_file_reader_index(int n);
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
    void set_convolution_mode(const bool enable);
    void set_divide_convolution_mode(const bool value);
    void set_fast_pipe(bool value);
    void toggle_renormalize(bool value);
    bool is_raw_mode();
    void reset();
    void update_filter2d_view(bool);
    void disable_filter2d_view();
    void set_filter2d(bool);
    void set_filter2d_n1(int);
    void set_filter2d_n2(int);
    void cancel_filter2d();
    void set_time_transformation_size();
    void update_lens_view(bool value);
    void disable_lens_view();
    void update_raw_view(bool value);
    void disable_raw_view();
    void set_p_accu();
    void set_x_accu();
    void set_y_accu();
    void set_x_y();
    void set_q(int value);
    void set_q_acc();
    void set_composite_intervals();
    void set_composite_intervals_hsv_h_min();
    void set_composite_intervals_hsv_h_max();
    void set_composite_intervals_hsv_s_min();
    void set_composite_intervals_hsv_s_max();
    void set_composite_intervals_hsv_v_min();
    void set_composite_intervals_hsv_v_max();
    void set_composite_weights();
    void set_composite_auto_weights(bool value);
    void click_composite_rgb_or_hsv();
    void slide_update_threshold_h_min();
    void slide_update_threshold_h_max();
    void slide_update_threshold_s_min();
    void slide_update_threshold_s_max();
    void slide_update_threshold_v_min();
    void slide_update_threshold_v_max();
    void actualize_frequency_channel_s();
    void actualize_frequency_channel_v();
    void actualize_checkbox_h_gaussian_blur();
    void actualize_kernel_size_blur();
    void set_p(int value);
    void increment_p();
    void decrement_p();
    void set_wavelength(double value);
    void set_z(double value);
    void increment_z();
    void decrement_z();
    void set_z_step(double value);
    void set_space_transformation(QString value);
    void set_time_transformation(QString value);
    void toggle_time_transformation_cuts(bool checked);
    void cancel_stft_slice_view();
    void update_batch_size();
    void update_time_transformation_stride();
    void set_view_mode(QString value);
    void set_unwrapping_2d(const bool value);
    void set_accumulation(bool value);
    void set_accumulation_level(int value);
    void set_contrast_mode(bool value);
    void set_auto_contrast();
    void set_contrast_min(double value);
    void set_contrast_max(double value);
    void invert_contrast(bool value);
    void set_auto_refresh_contrast(bool value);
    void set_log_scale(bool value);
    void set_fft_shift(bool value);
    void set_composite_area();
    void update_convo_kernel(const QString& value);
    void set_record_frame_step(int value);

    /*!
     * \brief Sets the start stop buttons object accessibility
     *
     * \param value accessibility
     */
    void set_start_stop_buttons(bool value);

    /*! \brief Opens file explorer to let the user chose the file he wants to import */
    void import_browse_file();

    /*!
     * \brief Creates an input file to gather data from it.
     *
     * \param filename the chosen file
     */
    void import_file(const QString& filename);

    /*! \brief Sets ui values and constraints + launch FileReadWroker */
    void init_holovibes_import_mode();

    /*! \brief Setups attributes for launching and launchs the imported file*/
    void import_start();
    /*! \brief Reset ui and stop holovibes' compute worker and file read worker */
    void import_stop();

    /*! \brief handle the ui input fps */
    void import_start_spinbox_update();

    /*! \brief handle the ui output fps */
    void import_end_spinbox_update();

    /*! \brief change the focused windows */
    void change_window();

    /*! \brief reloads .ini file that store program's state */
    void reload_ini();

    /*! \brief Saves the current state of holovibes in .ini file */
    void write_ini();
    void set_classic();
    void set_night();
    void rotateTexture();
    void flipTexture();
    void display_reticle(bool value);
    void reticle_scale(double value);

    void browse_record_output_file();
    void set_nb_frames_mode(bool value);
    void set_record_mode(const QString& value);
    void set_record_file_extension(const QString& value);
    void stop_record();
    void record_finished(RecordMode record_mode);
    void start_record();

    void browse_batch_input();

    void activeSignalZone();
    void activeNoiseZone();
    void start_chart_display();
    void stop_chart_display();
    /*! \} */

#pragma endregion
    /* ---------- */
  signals:
    void synchronize_thread_signal(std::function<void()> f);
#pragma region Protected / Private Methods
  protected:
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    void set_raw_mode();
    void set_holographic_mode();
    void set_computation_mode();
    void set_camera_timeout();
    void change_camera(CameraKind c);
    void display_error(std::string msg);
    void display_info(std::string msg);
    void open_file(const std::string& path);
    void load_ini(const std::string& path);

    /*!
     * \brief Saves the current state of holovibes
     *
     * \param path the location of the .ini file saved
     */
    void save_ini(const std::string& path);
    void cancel_time_transformation_cuts();
    void createPipe();
    void createHoloWindow();

    /*! \brief Closes all the displayed windows */
    void close_windows();
    void close_critical_compute();
    void remove_infos();
    void pipe_refresh();
    void set_auto_contrast_cuts();

    // Change the value without triggering any signals
    void QSpinBoxQuietSetValue(QSpinBox* spinBox, int value);
    void QSliderQuietSetValue(QSlider* slider, int value);
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

    uint window_max_size = 768;
    uint time_transformation_cuts_window_max_size = 512;
    uint auxiliary_window_max_size = 512;

    float displayAngle = 0.f;
    float xzAngle = 0.f;
    float yzAngle = 0.f;

    int displayFlip = 0;
    int xzFlip = 0;
    int yzFlip = 0;

    bool is_enabled_camera_ = false;
    double z_step_ = 0.005f;

    bool is_recording_ = false;
    unsigned record_frame_step_ = 512;
    RecordMode record_mode_ = RecordMode::RAW;

    std::string default_output_filename_;
    std::string record_output_directory_;
    std::string file_input_directory_;
    std::string batch_input_directory_;

    CameraKind kCamera = CameraKind::NONE;
    ImportType import_type_ = ImportType::None;
    QString last_img_type_ = "Magnitude";

    size_t auto_scale_point_threshold_ = 100;
    ushort theme_index_ = 0;

    // Shortcuts (initialized in constructor)
    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;

#pragma endregion
};
} // namespace gui
} // namespace holovibes
