/*! \file
 *
 * \brief Qt main class containing the GUI.
 */
#pragma once

// without namespace
#include "tools.hh"
#include "json.hh"
using json = ::nlohmann::json;

// namespace camera
#include "camera_exception.hh"

// namespace holovibes
#include "holovibes.hh"
#include "custom_exception.hh"

// namespace gui
#include "HoloWindow.hh"

#include "import_panel.hh"
#include "export_panel.hh"

#include "user_interface_descriptor.hh"

// Suppress all warnings in this auto-generated file
#pragma warning(push, 0)
#include "ui_mainwindow.h"
#pragma warning(pop)

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

  public:
    /*! \brief Set keyboard shortcuts, set visibility and load default values from holovibes.ini.
     *
     * \param parent Qt parent (should be null as it is the GUI hierarchy top class)
     */
    MainWindow(QWidget* parent = 0);
    ~MainWindow();

    void notify() override;
    void notify_error(const std::exception& e) override;

    /*! \brief Creates the windows for processed image output */
    void create_holo_window();

    /*! \brief Closes all the displayed windows */
    void close_windows();

    /*! \brief Stops critical compute */
    void close_critical_compute();

    /*! \brief Start the import process */
    void start_import(QString filename);

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

    void configure_holovibes();
    void browse_import_ini();
    void browse_export_ini();
    void reload_ini(QString filename);
    void reload_ini();
    void write_ini(QString filename);
    void write_ini();

    void configure_camera();
    void camera_none();
    void camera_adimec();
    void camera_ids();
    void camera_phantom();
    void camera_bitflow_cyton();
    void camera_hamamatsu();
    void camera_xiq();
    void camera_xib();

    /*! \brief Opens the credit display */
    void credits();
    void documentation();

    /*! \brief Changes the theme of the ui */
    void set_classic();
    void set_night();

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

    void refresh_view_mode();

    bool need_refresh(const std::string& last_type, const std::string& new_type);
    void set_composite_values();

    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_image_type(const QString& value);

    /*! \brief Changes the focused windows */
    void change_window();

  signals:
    /*! \brief TODO: comment
     *
     * \param f
     */
    void synchronize_thread_signal(std::function<void()> f);
#pragma region Protected / Private Methods
  public:
    /*! \brief Last call before the program is closed
     *
     * \param event Unused
     */
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    /*! \brief Sets camera frame timout */
    void set_camera_timeout();
  
  public:
    /*! \brief Changes camera
     *
     * \param c The new camera
     */
    void change_camera(CameraKind c);

    /*! \brief Setups program from .ini file
     *
     * \param path The path where the .ini file is
     */
    void load_ini(const std::string& path);

    /*! \brief Saves the current state of holovibes
     *
     * \param path The location of the .ini file saved
     */
    void save_ini(const std::string& path);

#pragma endregion
/* ---------- */
#pragma region Fields

    Ui::MainWindow* ui_;
    // ComputeDescriptor& cd_;
    std::vector<Panel*> panels_;

    QSpinBox* start_spinbox;
    QSpinBox* end_spinbox;

    // Additional attributs
    ushort theme_index_ = 0;

#pragma endregion
};
} // namespace gui
} // namespace holovibes
