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

    Ui::MainWindow* ui;
    Holovibes& holovibes_;
    ComputeDescriptor& cd_;
    std::unique_ptr<RawWindow> mainDisplay = nullptr;

    uint window_max_size = 768;
    uint auxiliary_window_max_size = 512;

    float displayAngle = 0.f; // V?
    float xzAngle = 0.f;      // V?
    float yzAngle = 0.f;      // V?

    int displayFlip = 0; // V?
    int xzFlip = 0;      // V?
    int yzFlip = 0;      // V?

    bool is_enabled_camera_ = false; // RIEN?IR?
    double z_step_ = 0.005f;         // IR

    CameraKind kCamera = CameraKind::NONE; // RIEN?IR?
    QString last_img_type_ = "Magnitude";  // V

    size_t auto_scale_point_threshold_ = 100; // RIEN
    ushort theme_index_ = 0;                  // RIEN

    // Shortcuts (initialized in constructor)
    QShortcut* z_up_shortcut_;    // ?
    QShortcut* z_down_shortcut_;  // ?
    QShortcut* p_left_shortcut_;  // ?
    QShortcut* p_right_shortcut_; // ?

  public slots:
    void on_notify();
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

    void refreshViewMode();

    /*! \brief Checks if we are currently in raw mode
     *
     * \return true if we are in raw mode, false otherwise
     */
    bool is_raw_mode();

    /*! \brief Resets the whole program in reload .ini file */
    void reset();

    /*! \brief Modifies view image type
     *
     * \param value The new image type
     */
    void set_view_image_type(const QString& value);

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

    /*! \brief Changes the theme of the ui */
    void set_classic();

    /*! \brief Changes the theme of the ui */
    void set_night();

  signals:
    /*! \brief TODO: comment
     *
     * \param f
     */
    void synchronize_thread_signal(std::function<void()> f);

  protected:
    /*! \brief Last call before the program is closed
     *
     * \param event Unused
     */
    virtual void closeEvent(QCloseEvent* event) override;

  public:
    /*! \brief Sets camera frame timout */
    void set_camera_timeout();

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

  private:
    std::vector<Panel*> panels_;
};
} // namespace gui
} // namespace holovibes
