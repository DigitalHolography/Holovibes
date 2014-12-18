#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <cmath>
# include <thread>
# include <iomanip>
# include <QMainWindow>
# include <QFileDialog>
# include <QShortcut>
# include <QMessageBox>
# include <QDesktopServices>
# include <boost/filesystem.hpp>
# include <boost/property_tree/ptree.hpp>
# include <boost/property_tree/ini_parser.hpp>
# include <gpib.h>
# include <camera_exception.hh>
# include "ui_main_window.h"
# include "holovibes.hh"
# include "pipeline.hh"
# include "compute_descriptor.hh"
# include "observer.hh"
# include "gui_gl_window.hh"
# include "gui_plot_window.hh"
# include "thread_recorder.hh"
# include "concurrent_deque.hh"

namespace gui
{
  class MainWindow : public QMainWindow, public holovibes::Observer
  {
    Q_OBJECT

  public:
    MainWindow(holovibes::Holovibes& holovibes, QWidget *parent = 0);
    ~MainWindow();

    void notify() override;

  public slots:
    // Menu
    void configure_holovibes();
    void gl_full_screen();
    void camera_none();
    void camera_edge();
    void camera_ids();
    void camera_ixon();
    void camera_pike();
    void camera_pixelfly();
    void camera_xiq();
    void configure_camera();
    void credits();

    // Image rendering
    void set_image_mode(bool value);
    void set_phase_number(int value);
    void set_p(int value);
    void increment_p();
    void decrement_p();
    void set_wavelength(double value);
    void set_z(double value);
    void increment_z();
    void decrement_z();
    void set_z_step(double value);
    void set_algorithm(QString value);

    // View
    void set_view_mode(QString value);
    void set_contrast_mode(bool value);
    void set_auto_contrast();
    void set_contrast_min(double value);
    void set_contrast_max(double value);
    void set_log_scale(bool value);
    void set_shifted_corners(bool value);

    // Special
    void set_vibro_mode(bool value);
    void set_p_vibro(int value);
    void set_q_vibro(int value);
    void set_average_mode(bool value);
    void set_average_graphic();
    void browse_roi_file();
    void browse_roi_output_file();
    void save_roi();
    void load_roi();

    // Record
    void browse_file();
    void set_record();
    void cancel_record();
    void finish_record();
    void browse_batch_input();
    void batch_record();
    void batch_next_record();
    void average_record();
    void test_average_record();
    void cancel_average_record();

  protected:
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    void global_visibility(bool value);
    void camera_visible(bool value);
    void contrast_visible(bool value);
    void record_visible(bool value);
    void record_but_cancel_visible(bool value);
    void image_ratio_visible(bool value);
    void average_visible(bool value);
    void average_record_but_cancel_visible(bool value);
    void change_camera(holovibes::Holovibes::camera_type camera_type);
    void display_error(std::string msg);
    void display_info(std::string msg);
    void open_file(const std::string& path);
    void load_ini(const std::string& path);
    void save_ini(const std::string& path);

  private:
    Ui::MainWindow ui;
    holovibes::Holovibes& holovibes_;
    GuiGLWindow* gl_window_;
    bool is_direct_mode_;
    bool is_enabled_camera_;
    bool is_enabled_average_;
    double z_step_;
    holovibes::Holovibes::camera_type camera_type_;

    PlotWindow* plot_window_;

    ThreadRecorder* record_thread_;
    QTimer average_record_timer_;
    unsigned int nb_frames_;

    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;
    QShortcut* gl_full_screen_;
    QShortcut* gl_normal_screen_;
  };
}

#endif /* !MAIN_WINDOW_HH_ */