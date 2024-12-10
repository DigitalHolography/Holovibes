/*! \file
 *
 * \brief Qt main class containing the GUI.
 */
#pragma once

// without namespace
#include "tools.hh"

// namespace camera
#include "camera_exception.hh"

#include "enum_camera_kind.hh"
#include "enum_theme.hh"

// namespace holovibes
#include "custom_exception.hh"

// panel struct
#include "import_panel.hh"
#include "export_panel.hh"

#include "lightui.hh"

// Suppress all warnings in this auto-generated file
#pragma warning(push, 0)

#include "ui_mainwindow.h"
#pragma warning(pop)

Q_DECLARE_METATYPE(std::function<void()>)

namespace holovibes::api
{
class Api;
}

namespace holovibes::gui
{
/*! \class MainWindow
 *
 * \brief Main class of the GUI. It regroup most of the Qt slots used for user actions.
 *
 * These slots are divided into several sections:
 *
 * * Menu: every action in the menu (e-g: configuration of .json, camera selection ...).
 * * Image rendering: #img, p, z, lambda ...
 * * View: log scale, shifted corner, contrast ...
 * * Special: image ratio, Chart plot ...
 * * Export: record of raw frames, Chart file ...
 * * Import : making a file of raw data the image source
 * * Info : Various runtime informations on the program's state
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

  public:
    /*! \brief Set keyboard shortcuts, set visibility and load default values from corresponding .json files.
     *
     * \param parent Qt parent (should be null as it is the GUI hierarchy top class)
     */
    MainWindow(QWidget* parent = 0);
    ~MainWindow();

    void notify();
    void notify_error(const std::exception& e);

    // Might be removed because all parameters can be accessed in UserInterfaceDescriptor
    friend class AdvancedSettingsWindow;

    /*! \brief Stops critical compute */
    void close_critical_compute();

    /*! \brief Start the import process */
    void start_import(QString filename);

    void set_preset_file_on_gpu();

    Ui::MainWindow* get_ui();

    uint window_max_size = 768;
    uint auxiliary_window_max_size = 512;

  public slots:
    void on_notify();
    /*! \brief Give a function to execute to the main thread with a signal
     *
     * \param f the function to execute
     */
    void synchronize_thread(std::function<void()> f);

    void browse_import_ini();
    void browse_export_ini();
    void reload_ini();
    void reload_ini(const std::string& filename);
    void write_compute_settings();
    void open_advanced_settings();
    void reset_settings();

    void configure_camera();
    void camera_none();
    void camera_adimec();
    void camera_ids();
    void camera_phantom();
    void camera_bitflow_cyton();
    void camera_hamamatsu();
    void camera_xiq();
    void camera_xib();
    void camera_opencv();
    void camera_ametek_s991_coaxlink_qspf_plus();
    void camera_ametek_s711_coaxlink_qspf_plus();
    void auto_detection_phantom();
    void camera_euresys_egrabber();
    void camera_alvium();

    void camera_adimec_settings();
    void camera_ids_settings();
    void camera_phantom_settings();
    void camera_bitflow_cyton_settings();
    void camera_hamamatsu_settings();
    void camera_xiq_settings();
    void camera_xib_settings();
    void camera_opencv_settings();
    void camera_ametek_s991_coaxlink_qspf_plus_settings();
    void camera_ametek_s711_coaxlink_qspf_plus_settings();
    void camera_euresys_egrabber_settings();
    void camera_alvium_settings();

    /*! \brief Opens the credit display */
    void credits();
    void documentation();

    /*! \brief Changes the theme of the ui */
    void set_classic();
    void set_night();
    void set_theme(const Theme theme);

    /*! \brief Resize windows if one layout is toggled. */
    void layout_toggled();

    /*
     * Lots of these methods stick to the following scheme:
     *
     * * Get pipe
     * * Set visibility to false
     * * Check if value is correct/into slot specific bounds
     * * Update a value in FrameDescriptor of the holovibes object
     * * Request a pipe refresh
     * * Set visibility to true
     */
    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_image_type(const QString& value);

    /*! \brief Changes the focused windows */
    void change_window(int index);

    void shift_screen();

    /**
     * @brief Open light ui panel and hide this one
     */
    void open_light_ui();

    /**
     * @brief Sets preset for given usage.
     */
    void set_preset();

    /**
     * @brief Sets preset given a preset file.
     */
    void set_preset(std::filesystem::path file);

  signals:
    /*! \brief TODO: comment
     *
     * \param f
     */
    void synchronize_thread_signal(std::function<void()> f);

  public:
    /*! \brief Last call before the program is closed
     *
     * \param event Unused
     */
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    /*! \brief Setups gui from .json file */
    void load_gui();
    void save_gui();

    /**
     * \brief Small helper function that just writes tooltips.
     * Note : the only affected locations should be drop-down menus, or 'comboBox'.
     */
    void init_tooltips();

  public:
    /*! \brief Changes camera
     *
     * \param c The new camera
     */
    void change_camera(CameraKind c);

    Ui::MainWindow* ui_;
    std::vector<Panel*> panels_;
    std::shared_ptr<LightUI> light_ui_;
    api::Api& api_;

    // Additional attributes
    Theme theme_ = Theme::Dark;
    bool save_cs = true;

  private:
    void enable_notify();
    void disable_notify();

    bool notify_enabled_ = true;

    Subscriber<bool> acquisition_started_subscriber_;
    Subscriber<bool> acquisition_finished_subscriber_;
    bool acquisition_finished_notification_received;
    Subscriber<bool> set_preset_subscriber_;
};
} // namespace holovibes::gui
