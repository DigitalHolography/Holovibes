#include "main_window.hh"

# define Z_STEP 0.01

namespace gui
{
  MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
    : QMainWindow(parent),
    holovibes_(holovibes),
    gl_window_(nullptr),
    is_direct_mode_(true),
    is_enabled_camera_(false),
    record_thread_(nullptr),
    z_step_(Z_STEP)
  {
    ui.setupUi(this);

    // FIXME (it will be when loading a camera from ini file)
    camera_visible(false);
    record_visible(false);

    // Keyboard shortcuts
    z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
    z_up_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_up_shortcut_, SIGNAL(activated()), this, SLOT(increment_z()));

    z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
    z_down_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_down_shortcut_, SIGNAL(activated()), this, SLOT(decrement_z()));

    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    p_left_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));

    if (is_direct_mode_)
      global_visibility(false);

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

    QComboBox* view_mode = findChild<QComboBox*>("viewModeComboBox");

    if (cd.view_mode == holovibes::ComputeDescriptor::MODULUS)
      view_mode->setCurrentIndex(0);
    else if (cd.view_mode == holovibes::ComputeDescriptor::SQUARED_MODULUS)
      view_mode->setCurrentIndex(1);
    else if (cd.view_mode == holovibes::ComputeDescriptor::ARGUMENT)
      view_mode->setCurrentIndex(2);
    else
      view_mode->setCurrentIndex(0);

    QCheckBox* log_scale = findChild<QCheckBox*>("logScaleCheckBox");
    log_scale->setChecked(cd.log_scale_enabled);

    QCheckBox* shift_corners = findChild<QCheckBox*>("shiftCornersCheckBox");
    shift_corners->setChecked(cd.shift_corners_enabled);

    QCheckBox* contrast = findChild<QCheckBox*>("contrastCheckBox");
    contrast->setChecked(cd.contrast_enabled);

    QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
    contrast_min->setValue(log10(cd.contrast_min));

    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
    contrast_max->setValue(log10(cd.contrast_max));

    QCheckBox* vibro = findChild<QCheckBox*>("vibrometryCheckBox");
    vibro->setChecked(cd.vibrometry_enabled);

    QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
    p_vibro->setValue(cd.pindex);

    QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
    q_vibro->setValue(cd.vibrometry_q);

    QCheckBox* average = findChild<QCheckBox*>("averageCheckBox");
    average->setChecked(cd.average_enabled);
  }

  void MainWindow::gl_full_screen()
  {
    gl_window_->full_screen();
  }

  void MainWindow::camera_ids()
  {
    change_camera(holovibes::Holovibes::IDS);
  }

  void MainWindow::camera_ixon()
  {
    change_camera(holovibes::Holovibes::IXON);
  }

  void MainWindow::camera_none()
  {
    delete gl_window_;
    gl_window_ = nullptr;
    if (!is_direct_mode_)
      holovibes_.dispose_compute();
    holovibes_.dispose_capture();
    camera_visible(false);
    record_visible(false);
    global_visibility(false);
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

  void MainWindow::credits()
  {
    display_info("Holovibes v0.4.3\n\n"
      "Scientists:\n"
      "Michael Atlan\n"
      "\n"
      "Developers:\n"
      "Jeffrey Bencteux\n"
      "Thomas Kostas\n"
      "Pierre Pagnoux\n");
  }

  void MainWindow::configure_camera()
  {
    open_file(boost::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
  }

  void MainWindow::set_image_mode(bool value)
  {
    if (is_enabled_camera_)
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
        gl_window_ = new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_capture_queue());
        is_direct_mode_ = true;

        global_visibility(false);
      }
      else
      {
        holovibes_.init_compute();
        gl_window_ = new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_output_queue());
        is_direct_mode_ = false;

        global_visibility(true);
      }
    }
  }

  void  MainWindow::set_phase_number(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      global_visibility(false);
      pipeline.request_update_n(value);
      global_visibility(true);
    }
  }

  void  MainWindow::set_p(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < (int)cd.nsamples)
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
      set_z(cd.zdistance + z_step_);
      QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
      z->setValue(cd.zdistance);
    }
  }

  void MainWindow::decrement_z()
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      set_z(cd.zdistance - z_step_);
      QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
      z->setValue(cd.zdistance);
    }
  }

  void MainWindow::set_z_step(double value)
  {
    z_step_ = value;
    QDoubleSpinBox* z_spinbox = findChild<QDoubleSpinBox*>("zSpinBox");
    z_spinbox->setSingleStep(value);
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

      set_contrast_min(contrast_min->value());
      set_contrast_max(contrast_max->value());

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
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.contrast_enabled)
      {
        if (cd.log_scale_enabled)
          cd.contrast_min = value;
        else
          cd.contrast_min = pow(10, value);

        pipeline.request_refresh();
      }
    }
  }

  void MainWindow::set_contrast_max(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.contrast_enabled)
      {
        if (cd.log_scale_enabled)
          cd.contrast_max = value;
        else
          cd.contrast_max = pow(10, value);

        pipeline.request_refresh();
      }
    }
  }

  void MainWindow::set_log_scale(bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.log_scale_enabled = value;

      if (cd.contrast_enabled)
      {
        QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
        QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
        set_contrast_min(contrast_min->value());
        set_contrast_max(contrast_max->value());
      }

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

      if (value < (int)cd.nsamples)
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

      if (value < (int)cd.nsamples)
      {
        holovibes_.get_compute_desc().vibrometry_q = value;
        pipeline.request_refresh();
      }
      else
        display_error("q param has to be between 0 and phase #");
    }
  }

  void MainWindow::set_average_mode(bool value)
  {
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
    GLWidget * gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    gl_widget->set_average_mode(value);

    if (!value)
    {
      holovibes_.get_compute_desc().average_enabled = false;
      pipeline.request_refresh();
    }
  }

  void MainWindow::browse_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("Record output file"), "C://", tr(""));

    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    path_line_edit->insert(filename);
  }

  void MainWindow::set_record()
  {
    global_visibility(false);
    record_but_cancel_visible(false);

    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");

    int nb_of_frames = nb_of_frames_spinbox->value();
    std::string path = path_line_edit->text().toUtf8();

    try
    {
      if (is_direct_mode_)
      {
        record_thread_ = new ThreadRecorder(
          holovibes_.get_capture_queue(),
          path,
          nb_of_frames,
          this);
      }
      else
      {
        record_thread_ = new ThreadRecorder(
          holovibes_.get_output_queue(),
          path,
          nb_of_frames,
          this);
      }

      connect(record_thread_, SIGNAL(finished()), this, SLOT(finish_record()));
      record_thread_->start();

      QPushButton* cancel_button = findChild<QPushButton*>("cancelPushButton");
      cancel_button->setDisabled(false);
    }
    catch (std::exception& e)
    {
      display_error(e.what());
    }
  }

  void MainWindow::cancel_record()
  {
    record_but_cancel_visible(true);

    if (record_thread_)
    {
      record_thread_->stop();
      display_info("Record canceled");

      if (!is_direct_mode_)
        global_visibility(true);
    }
  }

  void MainWindow::finish_record()
  {
    record_but_cancel_visible(true);

    QPushButton* cancel_button = findChild<QPushButton*>("cancelPushButton");
    cancel_button->setDisabled(true);
    delete record_thread_;
    record_thread_ = nullptr;
    display_info("Record has completed successfully");

    if (!is_direct_mode_)
      global_visibility(true);
  }

  void MainWindow::closeEvent(QCloseEvent* event)
  {
    if (gl_window_)
      gl_window_->close();
  }

  void MainWindow::global_visibility(bool value)
  {
    GroupBox* view = findChild<GroupBox*>("View");
    view->setDisabled(!value);

    GroupBox* special = findChild<GroupBox*>("Vibrometry");
    special->setDisabled(!value);

    QLabel* phase_number_label = findChild<QLabel*>("PhaseNumberLabel");
    phase_number_label->setDisabled(!value);

    QSpinBox* phase_nb = findChild<QSpinBox*>("phaseNumberSpinBox");
    phase_nb->setDisabled(!value);

    QLabel* p_label = findChild<QLabel*>("pLabel");
    p_label->setDisabled(!value);

    QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
    p->setDisabled(!value);

    QLabel* wavelength_label = findChild<QLabel*>("wavelengthLabel");
    wavelength_label->setDisabled(!value);

    QDoubleSpinBox* wavelength = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
    wavelength->setDisabled(!value);

    QLabel* z_label = findChild<QLabel*>("zLabel");
    z_label->setDisabled(!value);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setDisabled(!value);

    QLabel* z_step_label = findChild<QLabel*>("zStepLabel");
    z_step_label->setDisabled(!value);

    QDoubleSpinBox* z_step = findChild<QDoubleSpinBox*>("zStepDoubleSpinBox");
    z_step->setDisabled(!value);

    QLabel* algorithm_label = findChild<QLabel*>("algorithmLabel");
    algorithm_label->setDisabled(!value);

    QComboBox* algorithm = findChild<QComboBox*>("algorithmComboBox");
    algorithm->setDisabled(!value);
  }

  void MainWindow::camera_visible(bool value)
  {
    is_enabled_camera_ = value;
    gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("ImageRendering");
    image_rendering->setDisabled(!value);
    QAction* settings = findChild<QAction*>("actionSettings");
    settings->setDisabled(!value);
  }

  void MainWindow::record_visible(bool value)
  {
    gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("Record");
    image_rendering->setDisabled(!value);
  }

  void MainWindow::record_but_cancel_visible(bool value)
  {
    QLabel* nb_of_frames_label = findChild<QLabel*>("numberOfFramesLabel");
    nb_of_frames_label->setDisabled(!value);
    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    nb_of_frames_spinbox->setDisabled(!value);
    QLabel* output_file_label = findChild<QLabel*>("outputFileLabel");
    output_file_label->setDisabled(!value);
    QPushButton* browse_button = findChild<QPushButton*>("browsePushButton");
    browse_button->setDisabled(!value);
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    path_line_edit->setDisabled(!value);
    QPushButton* record_button = findChild<QPushButton*>("recPushButton");
    record_button->setDisabled(!value);
  }

  void MainWindow::change_camera(holovibes::Holovibes::camera_type camera_type)
  {
    try
    {
      camera_visible(false);
      record_visible(false);
      global_visibility(false);
      delete gl_window_;
      gl_window_ = nullptr;
      holovibes_.dispose_compute();
      holovibes_.dispose_capture();
      holovibes_.init_capture(camera_type, 20);
      camera_visible(true);
      record_visible(true);
      set_image_mode(is_direct_mode_);
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

  void MainWindow::display_info(std::string msg)
  {
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
  }

  void MainWindow::open_file(const std::string& path)
  {
    QDesktopServices::openUrl(QUrl(QString::fromUtf8(path.c_str())));
  }
}