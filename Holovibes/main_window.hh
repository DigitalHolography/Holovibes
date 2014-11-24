#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <thread>
# include <QMainWindow>
# include <QFileDialog>
# include "ui_main_window.h"
# include "holovibes.hh"
# include "pipeline.hh"
# include "compute_descriptor.hh"
# include "observer.hh"
# include "gui_gl_window.hh"

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
    // Image rendering
    void set_image_mode(bool value);
    void set_phase_number(int value);
    void set_p(int value);
    void set_wavelength(double value);
    void set_z(double value);
    void set_algorithm(QString value);

    // View
    void set_view_mode(QString value);
    void set_auto_contrast();
    void set_contrast_min(double value);
    void set_contrast_max(double value);
    void set_log_scale(bool value);
    void set_shifted_corners(bool value);

    // Special
    void set_p_vibro(int value);
    void set_q_vibro(int value);

    // Record
    void browse_file();
    void set_record();

  private:
    Ui::MainWindow ui;
    holovibes::Holovibes& holovibes_;
    GuiGLWindow* gl_window_;
    bool is_direct_mode_;

    void enable();
    void disable();

    //Debug
    template <typename T>
    void print_parameter(std::string name, T value);
  };
}

#endif /* !MAIN_WINDOW_HH_ */