#include "main_window.hh"

#define GLOBAL_INI_PATH "holovibes.ini"

namespace gui
{
  MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
    : QMainWindow(parent),
    holovibes_(holovibes),
    gl_window_(nullptr),
    is_direct_mode_(true),
    is_enabled_camera_(false),
    z_step_(0.01f),
    camera_type_(holovibes::Holovibes::NONE),
    record_thread_(nullptr)
  {
    ui.setupUi(this);

    camera_visible(false);
    record_visible(false);

    load_ini("holovibes.ini");

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
    p->setMaximum(cd.nsamples - 1);

    QDoubleSpinBox* lambda = findChild<QDoubleSpinBox*>("wavelengthSpinBox");
    lambda->setValue(cd.lambda * 1.0e9f);

    QDoubleSpinBox* z = findChild<QDoubleSpinBox*>("zSpinBox");
    z->setValue(cd.zdistance);

    QDoubleSpinBox* z_step = findChild<QDoubleSpinBox*>("zStepDoubleSpinBox");
    z_step->setValue(z_step_);

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
    /* Autocontrast values depends on log_scale option. */
    if (cd.log_scale_enabled)
      contrast_min->setValue(cd.contrast_min);
    else
      contrast_min->setValue(log10(cd.contrast_min));

    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
    if (cd.log_scale_enabled)
      contrast_max->setValue(cd.contrast_max);
    else
      contrast_max->setValue(log10(cd.contrast_max));

    QCheckBox* vibro = findChild<QCheckBox*>("vibrometryCheckBox");
    vibro->setChecked(cd.vibrometry_enabled);

    QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
    p_vibro->setValue(cd.pindex);
    p_vibro->setMaximum(cd.nsamples - 1);

    QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
    q_vibro->setValue(cd.vibrometry_q);

    QCheckBox* average = findChild<QCheckBox*>("averageCheckBox");
    average->setChecked(cd.average_enabled);
  }

  void MainWindow::configure_holovibes()
  {
    open_file(boost::filesystem::current_path().generic_string() + "/" + GLOBAL_INI_PATH);
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

  void MainWindow::camera_edge()
  {
    change_camera(holovibes::Holovibes::EDGE);
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
    display_info("Holovibes v0.5.2\n\n"
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
      notify();
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
        display_error("p param has to be between 0 and n");
    }
  }

  void MainWindow::increment_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex < cd.nsamples)
      {
        cd.pindex++;
        notify();
        pipeline.request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");
    }
  }

  void MainWindow::decrement_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex >= 0)
      {
        cd.pindex--;
        notify();
        pipeline.request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");
    }
  }

  void  MainWindow::set_wavelength(double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.lambda = static_cast<float>(value) * 1.0e-9f;
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
    QLabel* min_label = findChild<QLabel*>("minLabel");
    QLabel* max_label = findChild<QLabel*>("maxLabel");
    QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");

    min_label->setDisabled(!value);
    max_label->setDisabled(!value);
    contrast_min->setDisabled(!value);
    contrast_max->setDisabled(!value);

    if (!is_direct_mode_)
    {

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

      if (value < (int)cd.nsamples && value >= 0)
      {
        cd.pindex = value;
        notify();
        pipeline.request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");;
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

  void MainWindow::browse_roi_file()
  {
    QString filename = QFileDialog::getOpenFileName(this,
      tr("ROI output file"), "C://", tr("Ini files (*.ini)"));

    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    roi_output_line_edit->insert(filename);
  }

  void MainWindow::browse_roi_output_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("ROI output file"), "C://", tr("Ini files (*.ini)"));

    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    roi_output_line_edit->insert(filename);
  }

  void MainWindow::browse_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("Record output file"), "C://", tr("Raw files (*.raw);; All files (*)"));

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
    save_ini("holovibes.ini");

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
    if (camera_type != holovibes::Holovibes::NONE)
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
        camera_type_ = camera_type;
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


  void MainWindow::load_ini(const std::string& path)
  {
    boost::property_tree::ptree ptree;

    try
    {
      boost::property_tree::ini_parser::read_ini(path, ptree);
    }
    catch (std::exception& e)
    {
      std::ofstream os("holovibes.ini");
    }

    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

    if (!ptree.empty())
    {
      // Camera type
      int camera_type = ptree.get<int>("image_rendering.camera", 0);
      change_camera((holovibes::Holovibes::camera_type)camera_type);

      // Frame timeout
      int frame_timeout = ptree.get<int>("image_rendering.frame_timeout", camera::FRAME_TIMEOUT);
      camera::FRAME_TIMEOUT = frame_timeout;

      // Image rendering
      unsigned short phase_number = ptree.get<unsigned short>("image_rendering.phase_number", cd.nsamples);
      cd.nsamples = phase_number;

      unsigned short p_index = ptree.get<unsigned short>("image_rendering.p_index", cd.pindex);
      if (p_index >= 0 && p_index < cd.nsamples)
        cd.pindex = p_index;

      float lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);
      cd.lambda = lambda;

      float z_distance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);
      cd.zdistance = z_distance;

      float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
      if (z_step > 0.0f)
        z_step_ = z_step;

      int algorithm = ptree.get<int>("image_rendering.algorithm", cd.algorithm);
      cd.algorithm = (holovibes::ComputeDescriptor::fft_algorithm)algorithm;

      // View
      int view_mode = ptree.get<int>("view.view_mode", cd.view_mode);
      cd.view_mode = (holovibes::ComputeDescriptor::complex_view_mode)view_mode;

      bool log_scale_enabled = ptree.get<bool>("view.log_scale_enabled", cd.log_scale_enabled);
      cd.log_scale_enabled = log_scale_enabled;

      bool shift_corners_enabled = ptree.get<bool>("view.shift_corners_enabled", cd.shift_corners_enabled);
      cd.shift_corners_enabled = shift_corners_enabled;

      bool contrast_enabled = ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled);
      cd.contrast_enabled = contrast_enabled;

      float contrast_min = ptree.get<float>("view.contrast_min", cd.contrast_min);
      cd.contrast_min = contrast_min;

      float contrast_max = ptree.get<float>("view.contrast_max", cd.contrast_max);
      cd.contrast_max = contrast_max;
    }
  }

  void MainWindow::save_ini(const std::string& path)
  {
    boost::property_tree::ptree ptree;
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

    // Image rendering
    ptree.put("image_rendering.camera", camera_type_);
    ptree.put("image_rendering.frame_timeout", camera::FRAME_TIMEOUT);
    ptree.put("image_rendering.phase_number", cd.nsamples);
    ptree.put("image_rendering.p_index", cd.pindex);
    ptree.put("image_rendering.lambda", cd.lambda);
    ptree.put("image_rendering.z_distance", cd.zdistance);
    ptree.put("image_rendering.z_step", z_step_);
    ptree.put("image_rendering.algorithm", cd.algorithm);

    // View
    ptree.put("view.view_mode", cd.view_mode);
    ptree.put("view.log_scale_enabled", cd.log_scale_enabled);
    ptree.put("view.shift_corners_enabled", cd.shift_corners_enabled);
    ptree.put("view.contrast_enabled", cd.contrast_enabled);
    ptree.put("view.contrast_min", cd.contrast_min);
    ptree.put("view.contrast_max", cd.contrast_max);

    boost::property_tree::write_ini(path, ptree);
  }
}