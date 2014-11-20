#include "stdafx.h"
#include "main_window.hh"

namespace gui
{
  MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
    : QMainWindow(parent),
    holovibes_(holovibes),
    gl_window_(nullptr),
    is_direct_mode_(true)
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

    gl_window_ = new GuiGLWindow(QPoint(0, 0), 512, 512, holovibes_.get_capture_queue(), this);

    // Display default values
    notify();
  }

  MainWindow::~MainWindow()
  {
    delete gl_window_;
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
    holovibes_.dispose_compute();
    QPoint old_pos = gl_window_->pos();
    delete gl_window_;

    // If direct mode
    if (value)
    {
      gl_window_ = new GuiGLWindow(old_pos, 512, 512, holovibes_.get_capture_queue(), this);
      is_direct_mode_ = true;
    }
    else
    {
      holovibes_.init_compute();
      gl_window_ = new GuiGLWindow(old_pos, 512, 512, holovibes_.get_output_queue(), this);
      is_direct_mode_ = false;
    }
  }

  void  MainWindow::set_phase_number(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      pipeline.request_update_n(value);
    }
  }

  void  MainWindow::set_p(int value)
  {
    if (!is_direct_mode_)
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
  }

  void  MainWindow::set_wavelength(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.lambda = static_cast<float>(value);
      pipeline.request_refresh();
    }
  }

  void  MainWindow::set_z(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.zdistance = static_cast<float>(value);
      pipeline.request_refresh();
    }
  }

  void  MainWindow::set_algorithm(QString value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value == "1FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT1;
      else if (value == "2FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT2;

      pipeline.request_refresh();
    }
  }

  void MainWindow::set_view_mode(QString value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value == "magnitude")
        cd.view_mode = holovibes::ComputeDescriptor::MODULUS;
      else if (value == "squared magnitude")
        cd.view_mode = holovibes::ComputeDescriptor::SQUARED_MODULUS;
      else if (value == "argument")
        cd.view_mode = holovibes::ComputeDescriptor::ARGUMENT;
      else
        cd.view_mode = holovibes::ComputeDescriptor::MODULUS;

      pipeline.request_refresh();
    }
  }

  void MainWindow::set_auto_contrast()
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      pipeline.request_autocontrast();
    }
  }

  void MainWindow::set_contrast_min(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().contrast_min = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_contrast_max(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().contrast_max = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_log_scale(bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.log_scale_enabled = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_shifted_corners(bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().shift_corners_enabled = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_p_vibro(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().vibrometry_p = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_q_vibro(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().vibrometry_q = value;
      pipeline.request_refresh();
    }
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
    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    int nb_of_frames = nb_of_frames_spinbox->value();
    std::string path = path_line_edit->text().toUtf8();

    holovibes_.init_recorder(path, nb_of_frames);
    holovibes_.dispose_recorder();
  }

  template <typename T>
  void MainWindow::print_parameter(std::string name, T value)
  {
    std::cout << "Parameter " << name << " changed to " << std::boolalpha << value << std::endl;
  }
}