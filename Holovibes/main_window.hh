#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <cmath>
# include <thread>
# include <QMainWindow>
# include <QFileDialog>
# include <QShortcut>
# include <QMessageBox>
# include "ui_main_window.h"
# include "holovibes.hh"
# include "pipeline.hh"
# include "compute_descriptor.hh"
# include "observer.hh"
# include "gui_gl_window.hh"
# include "camera_exception.hh"

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
    void camera_none();
    void camera_ids();
    void camera_pike();
    void camera_pixelfly();
    void camera_xiq();

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

    // Record
    void browse_file();
    void set_record();

  protected:
    virtual void closeEvent(QCloseEvent* event) override;

  private:
    void enable();
    void disable();
    void change_camera(holovibes::Holovibes::camera_type camera_type);
    void display_error(std::string msg);

    //Debug
    template <typename T>
    void print_parameter(std::string name, T value);

  private:
    Ui::MainWindow ui;
    holovibes::Holovibes& holovibes_;
    GuiGLWindow* gl_window_;
    bool is_direct_mode_;

    QShortcut* z_up_shortcut_;
    QShortcut* z_down_shortcut_;
    QShortcut* p_left_shortcut_;
    QShortcut* p_right_shortcut_;
  };
}

#endif /* !MAIN_WINDOW_HH_ */