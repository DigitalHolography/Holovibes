#include "stdafx.h"
#include "main_window.hh"

namespace gui
{
  MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
    : QMainWindow(parent),
    holovibes_(holovibes)
  {
    ui.setupUi(this);

    // FIXME
    holovibes::ComputeDescriptor cd;
    cd.algorithm = holovibes::ComputeDescriptor::FFT1;
    cd.shift_corners_enabled = false;
    cd.pindex = 0;
    cd.nsamples = 4;
    cd.lambda = 536e-9f;
    cd.zdistance = 1.36f;

    holovibes_.set_compute_desc(cd);
    holovibes_.init_compute();

    // Display default values
    notify();
  }

  MainWindow::~MainWindow()
  {
  }

  void MainWindow::notify()
  {
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

    QSpinBox* phase_number = findChild<QSpinBox*>("phaseNumberSpinBox");
    phase_number->setValue(cd.nsamples);

    QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
    p->setValue(cd.pindex);

    QDoubleSpinBox* lambda = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
    lambda->setValue((double)cd.lambda);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setValue(cd.zdistance);

    QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");

    if (cd.algorithm == holovibes::ComputeDescriptor::FFT1)
      algorithm->setCurrentIndex(0);
    else if (cd.algorithm == holovibes::ComputeDescriptor::FFT2)
      algorithm->setCurrentIndex(1);
    else
      algorithm->setCurrentIndex(0);
  }

  void MainWindow::set_image_mode(bool value)
  {
    print_parameter("image mode", value);
  }

  void  MainWindow::set_phase_number(int value)
  {
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    pipeline.request_update_n(value);
  }

  void  MainWindow::set_p(int value)
  {
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
    
    if (value < cd.nsamples)
    {
      cd.pindex = value;
      pipeline.request_refresh();
    }
    else
      std::cout << "p param has to be between 0 and n" << "\n";
  }

  void  MainWindow::set_wavelength(double value)
  {
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
    cd.lambda = static_cast<float>(value);
    pipeline.request_refresh();
  }

  void  MainWindow::set_z(double value)
  {
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
    cd.zdistance = static_cast<float>(value);
    pipeline.request_refresh();
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
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    holovibes_.get_compute_desc().shift_corners_enabled = value;
    pipeline.request_refresh();
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