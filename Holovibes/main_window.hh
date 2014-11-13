#ifndef MAIN_WINDOW_HH_
# define MAIN_WINDOW_HH_

# include <QMainWindow>
# include "ui_main_window.h"

namespace gui
{
  class MainWindow : public QMainWindow
  {
    Q_OBJECT

  public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();

  public slots:
    // Image rendering
    void set_image_mode(bool value);
    void set_phase_number(int value);
    void set_p(int value);
    void set_wavelength(double value);
    void set_z(double value);
    void set_algorithm(QString value);

    // View
    void set_auto_contrast();
    void set_contrast_min(double value);
    void set_contrast_max(double value);
    void set_log_scale(bool value);
    void set_shifted_corners(bool value);

    // Special
    void set_p_vibro(int value);
    void set_q_vibro(int value);

    // Record
    void set_number_of_frames(int value);
    void set_record();

  private:
    Ui::MainWindow ui;

    //Debug
    template <typename T>
    void print_parameter(std::string name, T value);
  };
}

#endif /* !MAIN_WINDOW_HH_ */