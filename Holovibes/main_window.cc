#include "stdafx.h"
#include "main_window.hh"

namespace gui
{
  MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
  {
    ui.setupUi(this);
  }

  MainWindow::~MainWindow()
  {
  }

  void MainWindow::set_image_mode(bool value)
  {
    print_parameter("image mode", value);
  }

  void  MainWindow::set_phase_number(int value)
  {
    print_parameter("phase number", value);
  }

  void  MainWindow::set_p(int value)
  {
    print_parameter("p", value);
  }

  void  MainWindow::set_wavelength(double value)
  {
    print_parameter("wavelength", value);
  }

  void  MainWindow::set_z(double value)
  {
    print_parameter("z", value);
  }

  void  MainWindow::set_algorithm(QString value)
  {
    print_parameter("p", qPrintable(value));
  }

  void MainWindow::set_auto_contrast()
  {
    print_parameter("auto contrast", "enabled");
  }

  void MainWindow::set_contrast_min(double value)
  {
    print_parameter("contrast min", value);
  }

  void MainWindow::set_contrast_max(double value)
  {
    print_parameter("contrast max", value);
  }

  void MainWindow::set_log_scale(bool value)
  {
    print_parameter("log scale", value);
  }

  void MainWindow::set_shifted_corners(bool value)
  {
    print_parameter("shifted corners", value);
  }

  void MainWindow::set_p_vibro(int value)
  {
    print_parameter("p vibrometry", value);
  }

  void MainWindow::set_q_vibro(int value)
  {
    print_parameter("q vibrometry", value);
  }

  void MainWindow::set_number_of_frames(int value)
  {
    print_parameter("number of frames", value);
  }

  void MainWindow::browse_file()
  {
    QString filename = QFileDialog::getOpenFileName(this,
      tr("Record output file"), "C://", tr(""));

    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    path_line_edit->insert(filename);
  }

  void MainWindow::set_record()
  {
    print_parameter("record", "enabled");

    QProgressBar* record_progress_bar = findChild<QProgressBar*>("recordProgressBar");

    for (int i = 0; i < 100; ++i)
    {
      record_progress_bar->setValue(i);
      std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
  }

  template <typename T>
  void MainWindow::print_parameter(std::string name, T value)
  {
    std::cout << "Parameter " << name << " changed to " << std::boolalpha << value << std::endl;
  }
}