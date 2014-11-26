#include "main_window.hh"

# define Z_STEP 0.01

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
    
    // Keyboard shortcuts
    z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
    connect(z_up_shortcut_, SIGNAL(activated()), this, SLOT(increment_z()));

    z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
    connect(z_down_shortcut_, SIGNAL(activated()), this, SLOT(decrement_z()));

    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));

    if (is_direct_mode_)
      disable();

    // Display default values
    notify();
  }

  MainWindow::~MainWindow()
  {
    holovibes_.dispose_compute();
    holovibes_.dispose_capture();

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
    lambda->setValue(cd.lambda * 1.0e9f);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setValue(cd.zdistance);

    QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");

    if (cd.algorithm == holovibes::ComputeDescriptor::FFT1)
      algorithm->setCurrentIndex(0);
    else if (cd.algorithm == holovibes::ComputeDescriptor::FFT2)
      algorithm->setCurrentIndex(1);
    else
      algorithm->setCurrentIndex(0);

    QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
    contrast_min->setValue(log10(cd.contrast_min));

    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
    contrast_max->setValue(log10(cd.contrast_max));

    QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
    p_vibro->setValue(cd.pindex);

    QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
    q_vibro->setValue(cd.vibrometry_q);
  }

  void MainWindow::gl_full_screen()
  {
    gl_window_->full_screen();
  }

  void MainWindow::camera_ids()
  {
    change_camera(holovibes::Holovibes::IDS);
  }

  void MainWindow::camera_none()
  {
    delete gl_window_;
    gl_window_ = nullptr;
    if (!is_direct_mode_)
      holovibes_.dispose_compute();
    holovibes_.dispose_capture();
  }

  void MainWindow::camera_pike()
  {
    change_camera(holovibes::Holovibes::PIKE);
  }

  void MainWindow::camera_pixelfly()
  {
    change_camera(holovibes::Holovibes::PIXELFLY);
  }

  void MainWindow::camera_xiq()
  {
    change_camera(holovibes::Holovibes::XIQ);
  }

  void MainWindow::set_image_mode(bool value)
  {
    holovibes_.dispose_compute();
    QPoint pos(0, 0);
    unsigned int width = 512;
    unsigned int height = 512;

    if (gl_window_)
    {
      pos = gl_window_->pos();
      width = gl_window_->size().width();
      height = gl_window_->size().height();
    }

    delete gl_window_;

    // If direct mode
    if (value)
    {
      gl_window_ = new GuiGLWindow(pos, width, height, holovibes_.get_capture_queue());
      is_direct_mode_ = true;

      disable();
    }
    else
    {
      holovibes_.init_compute();
      gl_window_ = new GuiGLWindow(pos, width, height, holovibes_.get_output_queue());
      is_direct_mode_ = false;

      enable();
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
        // Synchronize with p_vibro
        QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
        p_vibro->setValue(value);

        cd.pindex = value;
        pipeline.request_refresh();
      }
      else
        std::cout << "p param has to be between 0 and n" << "\n";
    }
  }

  void MainWindow::increment_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex < cd.nsamples - 1)
      {
        cd.pindex++;
        notify();
        pipeline.request_refresh();
      }
    }
  }

  void MainWindow::decrement_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex > 0)
      {
        cd.pindex--;
        notify();
        pipeline.request_refresh();
      }
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

  void MainWindow::increment_z()
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      set_z(cd.zdistance + Z_STEP);
      QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
      z->setValue(cd.zdistance);
    }
  }

  void MainWindow::decrement_z()
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      set_z(cd.zdistance - Z_STEP);
      QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
      z->setValue(cd.zdistance);
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

  void MainWindow::set_contrast_mode(bool value)
  {
    if (!is_direct_mode_)
    {
      QLabel* min_label = findChild<QLabel*>("minLabel");
      QLabel* max_label = findChild<QLabel*>("maxLabel");
      QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
      QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");

      if (value)
      {
        min_label->setDisabled(false);
        max_label->setDisabled(false);
        contrast_min->setDisabled(false);
        contrast_max->setDisabled(false);
      }
      else
      {
        min_label->setDisabled(true);
        max_label->setDisabled(true);
        contrast_min->setDisabled(true);
        contrast_max->setDisabled(true);
      }

      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      cd.contrast_enabled = value;
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
      holovibes_.get_compute_desc().contrast_min = pow(10, value);
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_contrast_max(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes_.get_compute_desc().contrast_max = pow(10, value);
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

  void MainWindow::set_vibro_mode(bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      cd.vibrometry_enabled = value;
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_p_vibro(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < cd.nsamples)
      {
        // Synchronize with p
        QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
        p->setValue(value);

        cd.pindex = value;
        pipeline.request_refresh();
      }
      else
        display_error("p param has to be between 0 and phase #");;
    }
  }

  void MainWindow::set_q_vibro(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < cd.nsamples)
      {
        holovibes_.get_compute_desc().vibrometry_q = value;
        pipeline.request_refresh();
      }
      else
        display_error("q param has to be between 0 and phase #");
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
    if (!is_direct_mode_)
      disable();

    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    int nb_of_frames = nb_of_frames_spinbox->value();
    std::string path = path_line_edit->text().toUtf8();

    holovibes_.init_recorder(path, nb_of_frames);
    holovibes_.dispose_recorder();

    if (!is_direct_mode_)
      enable();
  }

  void MainWindow::closeEvent(QCloseEvent* event)
  {
    if (gl_window_)
      gl_window_->close();
  }

  void MainWindow::enable()
  {
    GroupBox* view = findChild<GroupBox*>("View");
    view->setDisabled(false);

    GroupBox* special = findChild<GroupBox*>("Vibrometry");
    special->setDisabled(false);

    QLabel* phase_number_label = findChild<QLabel*>("PhaseNumberLabel");
    phase_number_label->setDisabled(false);

    QSpinBox* phase_nb = findChild<QSpinBox*>("phaseNumberSpinBox");
    phase_nb->setDisabled(false);

    QLabel* p_label = findChild<QLabel*>("pLabel");
    p_label->setDisabled(false);

    QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
    p->setDisabled(false);

    QLabel* wavelength_label = findChild<QLabel*>("wavelengthLabel");
    wavelength_label->setDisabled(false);

    QDoubleSpinBox* wavelength = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
    wavelength->setDisabled(false);

    QLabel* z_label = findChild<QLabel*>("zLabel");
    z_label->setDisabled(false);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setDisabled(false);

    QLabel* algorithm_label = findChild<QLabel*>("algorithmLabel");
    algorithm_label->setDisabled(false);

    QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
    algorithm->setDisabled(false);
  }

  void MainWindow::disable()
  {
    GroupBox* view = findChild<GroupBox*>("View");
    view->setDisabled(true);

    GroupBox* special = findChild<GroupBox*>("Vibrometry");
    special->setDisabled(true);

    QLabel* phase_number_label = findChild<QLabel*>("PhaseNumberLabel");
    phase_number_label->setDisabled(true);

    QSpinBox* phase_nb = findChild<QSpinBox*>("phaseNumberSpinBox");
    phase_nb->setDisabled(true);

    QLabel* p_label = findChild<QLabel*>("pLabel");
    p_label->setDisabled(true);

    QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
    p->setDisabled(true);

    QLabel* wavelength_label = findChild<QLabel*>("wavelengthLabel");
    wavelength_label->setDisabled(true);

    QDoubleSpinBox* wavelength = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
    wavelength->setDisabled(true);

    QLabel* z_label = findChild<QLabel*>("zLabel");
    z_label->setDisabled(true);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setDisabled(true);

    QLabel* algorithm_label = findChild<QLabel*>("algorithmLabel");
    algorithm_label->setDisabled(true);

    QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
    algorithm->setDisabled(true);
  }

  void MainWindow::change_camera(holovibes::Holovibes::camera_type camera_type)
  {
    try
    {
      holovibes_.dispose_capture();

      holovibes_.init_capture(camera_type, 20);
      set_image_mode(true);
    }
    catch (camera::CameraException& e)
    {
      display_error("[CAMERA]" + std::string(e.what()));
    }
    catch (std::exception& e)
    {
      display_error(e.what());
    }
  }

  void MainWindow::display_error(std::string msg)
  {
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Critical);
    msg_box.exec();
  }

  template <typename T>
  void MainWindow::print_parameter(std::string name, T value)
  {
    std::cout << "Parameter " << name << " changed to " << std::boolalpha << value << std::endl;
  }
}