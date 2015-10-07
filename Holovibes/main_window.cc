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
    is_enabled_average_(false),
    is_batch_img_(true),
    is_batch_interrupted_(false),
    z_step_(0.1f),
    camera_type_(holovibes::Holovibes::NONE),
    plot_window_(nullptr),
    record_thread_(nullptr),
    CSV_record_thread_(nullptr),
    file_index_(1),
    q_max_size_(20)
  {
    ui.setupUi(this);
    this->setWindowIcon(QIcon("icon1.ico"));

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

    autofocus_ctrl_c_shortcut_ = new QShortcut(tr("Ctrl+C"), this);
    autofocus_ctrl_c_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(autofocus_ctrl_c_shortcut_, SIGNAL(activated()), this, SLOT(request_autofocus_stop()));

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
    else if (cd.algorithm == holovibes::ComputeDescriptor::STFT)
      algorithm->setCurrentIndex(2);
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

    contrast_visible(cd.contrast_enabled);

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

    image_ratio_visible(cd.vibrometry_enabled);

    QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
    p_vibro->setValue(cd.pindex);
    p_vibro->setMaximum(cd.nsamples - 1);

    QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
    q_vibro->setValue(cd.vibrometry_q);
    q_vibro->setMaximum(cd.nsamples - 1);

    QCheckBox* average = findChild<QCheckBox*>("averageCheckBox");
    average->setChecked(is_enabled_average_);

    GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    if (gl_widget && is_enabled_average_)
      gl_widget->set_selection_mode(gui::eselection::AVERAGE);

    average_visible(is_enabled_average_);
  }

  void MainWindow::configure_holovibes()
  {
    open_file(boost::filesystem::current_path().generic_string() + "/" + GLOBAL_INI_PATH);
  }

  void MainWindow::gl_full_screen()
  {
    if (gl_window_)
      gl_window_->full_screen();
    else
      display_error("No camera selected");
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
    display_info("Holovibes v1.0.2\n\n"
      "Scientists:\n"
      "Michael Atlan\n"
      "\n"
      "Developers:\n"
      "Jeffrey Bencteux\n"
      "Thomas Kostas\n"
      "Pierre Pagnoux\n"
      "\n"
      "Eric Delanghe\n"
      "Arnaud Gaillard\n"
      "Geoffrey Le Gourrierec\n");
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

      notify();
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

        cd.pindex.exchange(value);
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
        ++(cd.pindex);
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
        --(cd.pindex);
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
      cd.lambda = static_cast<float>(value)* 1.0e-9f;
      pipeline.request_refresh();

      // Updating the GUI
      QLineEdit* boundary = findChild<QLineEdit*>("boundary");
      boundary->clear();
      boundary->insert(QString::number(holovibes_.get_boundary()));
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
      QSpinBox* phaseNumberSpinBox = findChild<QSpinBox*>("phaseNumberSpinBox");

      cd.nsamples = 2;
      if (value == "1FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT1;
      else if (value == "2FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT2;
      else if (value == "STFT")
      {
        cd.nsamples = 32;
        cd.algorithm = holovibes::ComputeDescriptor::STFT;
      }
      else
        assert(!"Unknow Algorithm.");

      phaseNumberSpinBox->setValue(cd.nsamples);
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

  void MainWindow::set_autofocus_mode()
  {
    GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    gl_widget->set_selection_mode(gui::eselection::AUTOFOCUS);

    float z_max = findChild<QDoubleSpinBox*>("zmaxDoubleSpinBox")->value();
    float z_min = findChild<QDoubleSpinBox*>("zminDoubleSpinBox")->value();
    unsigned int z_div = findChild<QSpinBox*>("zdivSpinBox")->value();
    unsigned int z_iter = findChild<QSpinBox*>("ziterSpinBox")->value();
    holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

    if (z_min < z_max)
    {
      desc.autofocus_z_min = z_min;
      desc.autofocus_z_max = z_max;
      desc.autofocus_z_div.exchange(z_div);
      desc.autofocus_z_iter = z_iter;

      connect(gl_widget, SIGNAL(autofocus_zone_selected(holovibes::Rectangle)), this, SLOT(request_autofocus(holovibes::Rectangle)),
        Qt::UniqueConnection);
    }
    else
      display_error("z min has to be strictly inferior to z max");
  }

  void MainWindow::request_autofocus(holovibes::Rectangle zone)
  {
    GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();
    holovibes::Pipeline& pipeline = holovibes_.get_pipeline();

    desc.autofocus_zone = zone;
    pipeline.request_autofocus();
    gl_widget->set_selection_mode(gui::eselection::ZOOM);
  }

  void MainWindow::request_autofocus_stop()
  {
    try
    {
      holovibes_.get_pipeline().request_autofocus_stop();
    }
    catch (std::runtime_error& e)
    {
      std::cerr << e.what() << std::endl;
    }
  }

  void MainWindow::set_contrast_mode(bool value)
  {
    QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
    contrast_visible(value);

    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.contrast_enabled.exchange(value);

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
      cd.log_scale_enabled.exchange(value);

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
      holovibes_.get_compute_desc().shift_corners_enabled.exchange(value);
      pipeline.request_refresh();
    }
  }

  void MainWindow::set_vibro_mode(bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::Pipeline& pipeline = holovibes_.get_pipeline();
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      image_ratio_visible(value);
      cd.vibrometry_enabled.exchange(value);
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
        cd.pindex.exchange(value);
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

      if (value < (int)cd.nsamples && value >= 0)
      {
        holovibes_.get_compute_desc().vibrometry_q.exchange(value);
        pipeline.request_refresh();
      }
      else
        display_error("q param has to be between 0 and phase #");
    }
  }

  void MainWindow::set_average_mode(bool value)
  {
    GLWidget * gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    if (value)
      gl_widget->set_selection_mode(gui::eselection::AVERAGE);
    else
      gl_widget->set_selection_mode(gui::eselection::ZOOM);
    is_enabled_average_ = value;

    // TODO
    average_visible(value);
  }

  void MainWindow::set_average_graphic()
  {
    delete plot_window_;
    holovibes_.get_pipeline().request_average(&holovibes_.get_average_queue());
    plot_window_ = new PlotWindow(holovibes_.get_average_queue(), "ROI Average");
  }

  void MainWindow::browse_roi_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("ROI output file"), "C://", tr("Ini files (*.ini)"));

    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    roi_output_line_edit->clear();
    roi_output_line_edit->insert(filename);
  }

  void MainWindow::browse_roi_output_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("ROI output file"), "C://", tr("Text files (*.txt);;CSV files (*.csv)"));

    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    roi_output_line_edit->clear();
    roi_output_line_edit->insert(filename);
  }

  void MainWindow::save_roi()
  {
    QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    std::string path = path_line_edit->text().toUtf8();

    if (path != "")
    {
      boost::property_tree::ptree ptree;
      const GLWidget& gl_widget = gl_window_->get_gl_widget();
      const holovibes::Rectangle& signal = gl_widget.get_signal_selection();
      const holovibes::Rectangle& noise = gl_widget.get_noise_selection();

      ptree.put("signal.top_left_x", signal.top_left.x);
      ptree.put("signal.top_left_y", signal.top_left.y);
      ptree.put("signal.bottom_right_x", signal.bottom_right.x);
      ptree.put("signal.bottom_right_y", signal.bottom_right.y);

      ptree.put("noise.top_left_x", noise.top_left.x);
      ptree.put("noise.top_left_y", noise.top_left.y);
      ptree.put("noise.bottom_right_x", noise.bottom_right.x);
      ptree.put("noise.bottom_right_y", noise.bottom_right.y);

      boost::property_tree::write_ini(path, ptree);
      display_info("Roi saved in " + path);
    }
    else
      display_error("Invalid path");
  }

  void MainWindow::load_roi()
  {
    QLineEdit* path_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    std::string path = path_line_edit->text().toUtf8();
    boost::property_tree::ptree ptree;
    GLWidget& gl_widget = gl_window_->get_gl_widget();

    try
    {
      boost::property_tree::ini_parser::read_ini(path, ptree);

      holovibes::Point2D signal_top_left;
      holovibes::Point2D signal_bottom_right;
      holovibes::Point2D noise_top_left;
      holovibes::Point2D noise_bottom_right;

      signal_top_left.x = ptree.get<int>("signal.top_left_x", 0);
      signal_top_left.y = ptree.get<int>("signal.top_left_y", 0);
      signal_bottom_right.x = ptree.get<int>("signal.bottom_right_x", 0);
      signal_bottom_right.y = ptree.get<int>("signal.bottom_right_y", 0);

      noise_top_left.x = ptree.get<int>("noise.top_left_x", 0);
      noise_top_left.y = ptree.get<int>("noise.top_left_y", 0);
      noise_bottom_right.x = ptree.get<int>("noise.bottom_right_x", 0);
      noise_bottom_right.y = ptree.get<int>("noise.bottom_right_y", 0);

      holovibes::Rectangle signal(signal_top_left, signal_bottom_right);
      holovibes::Rectangle noise(noise_top_left, noise_bottom_right);

      gl_widget.set_signal_selection(signal);
      gl_widget.set_noise_selection(noise);
      gl_widget.enable_selection();
    }
    catch (std::exception& e)
    {
      display_error("Couldn't load ini file\n" + std::string(e.what()));
    }
  }

  void MainWindow::browse_file()
  {
    QString filename = QFileDialog::getSaveFileName(this,
      tr("Record output file"), "C://", tr("Raw files (*.raw);; All files (*)"));

    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    path_line_edit->clear();
    path_line_edit->insert(filename);
  }

  void MainWindow::set_record()
  {
    global_visibility(false);
    record_but_cancel_visible(false);

    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");

    int nb_of_frames = nb_of_frames_spinbox->value();
    std::string path = path_line_edit->text().toUtf8();

    try
    {
      if (float_output_checkbox->isChecked() && !is_direct_mode_)
      {
        holovibes_.get_pipeline().request_float_output(path, nb_of_frames);

        global_visibility(true);
        record_but_cancel_visible(true);

        while (holovibes_.get_pipeline().is_requested_float_output())
          std::this_thread::yield();
        display_info("Record done");
      }
      else
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

        connect(record_thread_, SIGNAL(finished()), this, SLOT(finished_image_record()));
        record_thread_->start();

        QPushButton* cancel_button = findChild<QPushButton*>("cancelPushButton");
        cancel_button->setDisabled(false);
      }
    }
    catch (std::exception& e)
    {
      display_error(e.what());
      global_visibility(true);
      record_but_cancel_visible(true);
    }
  }

  void MainWindow::finished_image_record()
  {
    delete record_thread_;
    record_thread_ = nullptr;
    display_info("Record done");
    if (!is_direct_mode_)
      global_visibility(true);
    record_but_cancel_visible(true);
  }

  void MainWindow::average_record()
  {
    if (plot_window_)
    {
      plot_window_->stop_drawing();
      delete plot_window_;
      plot_window_ = nullptr;
      holovibes_.get_pipeline().request_refresh();
    }

    QSpinBox* nb_of_frames_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
    nb_frames_ = nb_of_frames_spin_box->value();
    QLineEdit* output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    std::string output_path = output_line_edit->text().toUtf8();

    CSV_record_thread_ = new ThreadCSVRecord(holovibes_,
      holovibes_.get_average_queue(),
      output_path,
      nb_frames_,
      this);
    connect(CSV_record_thread_, SIGNAL(finished()), this, SLOT(finished_average_record()));
    CSV_record_thread_->start();

    global_visibility(false);
    record_but_cancel_visible(false);
    average_record_but_cancel_visible(false);
    QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIStopPushButton");
    roi_stop_push_button->setDisabled(false);
  }

  void MainWindow::finished_average_record()
  {
    delete CSV_record_thread_;
    CSV_record_thread_ = nullptr;
    display_info("ROI record done");

    global_visibility(true);
    record_but_cancel_visible(true);
    average_record_but_cancel_visible(true);
    QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIStopPushButton");
    roi_stop_push_button->setDisabled(true);
  }

  void MainWindow::browse_batch_input()
  {
    QString filename = QFileDialog::getOpenFileName(this,
      tr("Batch input file"), "C://", tr("All files (*)"));

    QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("batchInputLineEdit");
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
  }

  void MainWindow::image_batch_record()
  {
    QLineEdit* output_path = findChild<QLineEdit*>("pathLineEdit");

    is_batch_img_ = true;
    is_batch_interrupted_ = false;
    batch_record(std::string(output_path->text().toUtf8()));
  }

  void MainWindow::csv_batch_record()
  {
    if (plot_window_)
    {
      plot_window_->stop_drawing();
      delete plot_window_;
      plot_window_ = nullptr;
      holovibes_.get_pipeline().request_refresh();
    }

    QLineEdit* output_path = findChild<QLineEdit*>("ROIOutputLineEdit");

    is_batch_img_ = false;
    is_batch_interrupted_ = false;
    batch_record(std::string(output_path->text().toUtf8()));
  }

  void MainWindow::batch_record(const std::string& path)
  {
    file_index_ = 1;
    struct stat buff;
    QLineEdit* batch_input_line_edit = findChild<QLineEdit*>("batchInputLineEdit");
    QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");

    std::string input_path = batch_input_line_edit->text().toUtf8();
    unsigned int frame_nb = frame_nb_spin_box->value();

    int status = load_batch_file(input_path.c_str());
    std::string formatted_path = format_batch_output(path, file_index_);

    if (status != 0)
      display_error("Couldn't load batch input file.");
    else if (path == "")
      display_error("Please provide an output file path.");
    else if (stat(formatted_path.c_str(), &buff) == 0)
      display_error("File: " + path + " already exists.");
    else
    {
      global_visibility(false);
      camera_visible(false);

      holovibes::Queue* q;

      if (is_direct_mode_)
        q = &holovibes_.get_capture_queue();
      else
        q = &holovibes_.get_output_queue();

      execute_next_block();

      if (is_batch_img_)
      {
        record_thread_ = new ThreadRecorder(*q, formatted_path, frame_nb, this);
        connect(record_thread_, SIGNAL(finished()), this, SLOT(batch_next_record()));
        record_thread_->start();
      }
      else
      {
        CSV_record_thread_ = new ThreadCSVRecord(holovibes_,
          holovibes_.get_average_queue(),
          formatted_path,
          frame_nb,
          this);
        connect(CSV_record_thread_, SIGNAL(finished()), this, SLOT(batch_next_record()));
        CSV_record_thread_->start();
      }

      ++file_index_;
    }
  }

  void MainWindow::batch_next_record()
  {
    if (!is_batch_interrupted_)
    {
      delete record_thread_;

      QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
      std::string path;

      if (is_batch_img_)
        path = findChild<QLineEdit*>("pathLineEdit")->text().toUtf8();
      else
        path = findChild<QLineEdit*>("ROIOutputLineEdit")->text().toUtf8();

      unsigned int frame_nb = frame_nb_spin_box->value();

      holovibes::Queue* q;

      if (is_direct_mode_)
        q = &holovibes_.get_capture_queue();
      else
        q = &holovibes_.get_output_queue();

      std::string output_filename = format_batch_output(path, file_index_);

      if (is_batch_img_)
      {
        record_thread_ = new ThreadRecorder(*q, output_filename, frame_nb, this);

        if (execute_next_block())
          connect(record_thread_, SIGNAL(finished()), this, SLOT(batch_next_record()));
        else
          connect(record_thread_, SIGNAL(finished()), this, SLOT(batch_finished_record()));

        record_thread_->start();
      }
      else
      {
        CSV_record_thread_ = new ThreadCSVRecord(holovibes_,
          holovibes_.get_average_queue(),
          output_filename,
          frame_nb,
          this);

        if (execute_next_block())
          connect(CSV_record_thread_, SIGNAL(finished()), this, SLOT(batch_next_record()));
        else
          connect(CSV_record_thread_, SIGNAL(finished()), this, SLOT(batch_finished_record()));

        CSV_record_thread_->start();
      }

      file_index_++;
    }
    else
    {
      batch_finished_record();
    }
  }

  void MainWindow::batch_finished_record()
  {
    delete record_thread_;
    record_thread_ = nullptr;
    delete CSV_record_thread_;
    CSV_record_thread_ = nullptr;
    file_index_ = 1;
    global_visibility(true);
    camera_visible(true);
    display_info("Batch record done");

    if (plot_window_)
    {
      plot_window_->stop_drawing();
      holovibes_.get_pipeline().request_average(&holovibes_.get_average_queue());
      plot_window_->start_drawing();
    }
  }

  void MainWindow::stop_image_record()
  {
    if (record_thread_)
    {
      record_thread_->stop();
      is_batch_interrupted_ = true;
    }
  }

  void MainWindow::stop_csv_record()
  {
    if (is_enabled_average_)
    {
      if (CSV_record_thread_)
      {
        CSV_record_thread_->stop();
        is_batch_interrupted_ = true;
      }
    }
  }

  void MainWindow::import_browse_file()
  {
    QString filename = QFileDialog::getOpenFileName(this,
      tr("import file"), "C://", tr("All files (*)"));

    QLineEdit* import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
    import_line_edit->clear();
    import_line_edit->insert(filename);
  }

  void MainWindow::import_file_stop(void)
  {
    change_camera(camera_type_);
  }

  void MainWindow::import_file()
  {
    QLineEdit* import_line_edit = findChild<QLineEdit*>("ImportPathLineEdit");
    QSpinBox* width_spinbox = findChild<QSpinBox*>("ImportWidthSpinBox");
    QSpinBox* height_spinbox = findChild<QSpinBox*>("ImportHeightSpinBox");
    QSpinBox* fps_spinbox = findChild<QSpinBox*>("ImportFpsSpinBox");
    QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
    QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");
    QComboBox* depth_spinbox = findChild<QComboBox*>("ImportDepthModeComboBox");
    QCheckBox* loop_checkbox = findChild<QCheckBox*>("ImportLoopCheckBox");
    QCheckBox* squared_checkbox = findChild<QCheckBox*>("ImportSquaredCheckBox");
    QComboBox* big_endian_checkbox = findChild<QComboBox*>("ImportEndianModeComboBox");

    std::string file_src = import_line_edit->text().toUtf8();

    holovibes::ThreadReader::FrameDescriptor frame_desc({
      width_spinbox->value(),
      height_spinbox->value(),
      // 0:depth = 8, 1:depth = 16
      depth_spinbox->currentIndex() + 1,
      (big_endian_checkbox->currentText() == QString("Big Endian") ? camera::endianness::BIG_ENDIAN : camera::endianness::LITTLE_ENDIAN),
    });

    camera_visible(false);
    record_visible(false);
    global_visibility(false);
    delete gl_window_;
    gl_window_ = nullptr;
    holovibes_.dispose_compute();
    holovibes_.dispose_capture();
    holovibes_.init_import_mode(
      file_src,
      frame_desc,
      loop_checkbox->isChecked(),
      fps_spinbox->value(),
      start_spinbox->value(),
      end_spinbox->value(),
      q_max_size_);
    camera_visible(true);
    record_visible(true);
    set_image_mode(is_direct_mode_);
  }

  void MainWindow::import_start_spinbox_update()
  {
    QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
    QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");

    if (start_spinbox->value() > end_spinbox->value())
      end_spinbox->setValue(start_spinbox->value());
  }

  void MainWindow::import_end_spinbox_update()
  {
    QSpinBox* start_spinbox = findChild<QSpinBox*>("ImportStartSpinBox");
    QSpinBox* end_spinbox = findChild<QSpinBox*>("ImportEndSpinBox");

    if (end_spinbox->value() < start_spinbox->value())
      start_spinbox->setValue(end_spinbox->value());
  }

  void MainWindow::closeEvent(QCloseEvent* event)
  {
    (void)event;
    save_ini("holovibes.ini");

    if (gl_window_)
      gl_window_->close();

    if (plot_window_)
      plot_window_->close();
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

    QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");
    float_output_checkbox->setDisabled(!value);

    QLineEdit* pixelSize = findChild<QLineEdit*>("pixelSize");
    pixelSize->setDisabled(!value);

    QLineEdit* boundary = findChild<QLineEdit*>("boundary");
    boundary->setDisabled(!value);
  }

  void MainWindow::camera_visible(bool value)
  {
    is_enabled_camera_ = value;
    gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("ImageRendering");
    image_rendering->setDisabled(!value);
    QAction* settings = findChild<QAction*>("actionSettings");
    settings->setDisabled(!value);
  }

  void MainWindow::contrast_visible(bool value)
  {
    QLabel* min_label = findChild<QLabel*>("minLabel");
    QLabel* max_label = findChild<QLabel*>("maxLabel");
    QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
    QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");

    min_label->setDisabled(!value);
    max_label->setDisabled(!value);
    contrast_min->setDisabled(!value);
    contrast_max->setDisabled(!value);
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

  void MainWindow::image_ratio_visible(bool value)
  {
    QLabel* p_label_vibro = findChild<QLabel*>("pLabelVibro");
    p_label_vibro->setDisabled(!value);
    QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
    p_vibro->setDisabled(!value);
    QLabel* q_label_vibro = findChild<QLabel*>("qLabelVibro");
    q_label_vibro->setDisabled(!value);
    QSpinBox* q_vibro = findChild<QSpinBox*>("qSpinBoxVibro");
    q_vibro->setDisabled(!value);
  }

  void MainWindow::average_visible(bool value)
  {
    QLabel* roi_file_label = findChild<QLabel*>("ROIFileLabel");
    roi_file_label->setDisabled(!value);
    QPushButton* roi_browse_button = findChild<QPushButton*>("ROIBrowseButton");
    roi_browse_button->setDisabled(!value);
    QLineEdit* roi_file_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    roi_file_line_edit->setDisabled(!value);
    QPushButton* save_roi_button = findChild<QPushButton*>("saveROIPushButton");
    save_roi_button->setDisabled(!value);
    QPushButton* load_roi_button = findChild<QPushButton*>("loadROIPushButton");
    load_roi_button->setDisabled(!value);
  }

  void MainWindow::average_record_but_cancel_visible(bool value)
  {
    QLabel* roi_output_file_label = findChild<QLabel*>("ROIOutputFileLabel");
    roi_output_file_label->setDisabled(!value);
    QPushButton* roi_output_push_button = findChild<QPushButton*>("ROIOutputPushButton");
    roi_output_push_button->setDisabled(!value);
    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    roi_output_line_edit->setDisabled(!value);
    QPushButton* roi_push_button = findChild<QPushButton*>("ROIPushButton");
    roi_push_button->setDisabled(!value);
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
        holovibes_.init_capture(camera_type, q_max_size_);
        camera_visible(true);
        record_visible(true);
        set_image_mode(is_direct_mode_);
        camera_type_ = camera_type;

        // Changing the gui
        QLineEdit* pixel_size = findChild<QLineEdit*>("pixelSize");
        pixel_size->clear();
        pixel_size->insert(QString::number(holovibes_.get_cam_frame_desc().pixel_size));

        QLineEdit* boundary = findChild<QLineEdit*>("boundary");
        boundary->clear();
        boundary->insert(QString::number(holovibes_.get_boundary()));
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
    gui::GroupBox	*image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
    gui::GroupBox	*view_group_box = findChild<gui::GroupBox*>("View");
    gui::GroupBox	*special_group_box = findChild<gui::GroupBox*>("Vibrometry");
    gui::GroupBox	*record_group_box = findChild<gui::GroupBox*>("Record");
    gui::GroupBox	*import_group_box = findChild<gui::GroupBox*>("Import");

    try
    {
      boost::property_tree::ini_parser::read_ini(path, ptree);
    }
    catch (std::exception& e)
    {
      std::cout << e.what() << std::endl;
    }

    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

    if (!ptree.empty())
    {
      // Queue max size
      q_max_size_ = ptree.get<int>("image_rendering.queue_size", q_max_size_);

      // Camera type
      int camera_type = ptree.get<int>("image_rendering.camera", 0);
      change_camera((holovibes::Holovibes::camera_type)camera_type);

      // Frame timeout
      int frame_timeout = ptree.get<int>("image_rendering.frame_timeout", camera::FRAME_TIMEOUT);
      camera::FRAME_TIMEOUT = frame_timeout;

      // Image rendering
      image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

      unsigned short phase_number = ptree.get<unsigned short>("image_rendering.phase_number", cd.nsamples);
      cd.nsamples = phase_number;

      unsigned short p_index = ptree.get<unsigned short>("image_rendering.p_index", cd.pindex);
      if (p_index >= 0 && p_index < cd.nsamples)
        cd.pindex.exchange(p_index);

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
      view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

      int view_mode = ptree.get<int>("view.view_mode", cd.view_mode);
      cd.view_mode = (holovibes::ComputeDescriptor::complex_view_mode)view_mode;

      bool log_scale_enabled = ptree.get<bool>("view.log_scale_enabled", cd.log_scale_enabled);
      cd.log_scale_enabled.exchange(log_scale_enabled);

      bool shift_corners_enabled = ptree.get<bool>("view.shift_corners_enabled", cd.shift_corners_enabled);
      cd.shift_corners_enabled.exchange(shift_corners_enabled);

      bool contrast_enabled = ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled);
      cd.contrast_enabled.exchange(contrast_enabled);

      float contrast_min = ptree.get<float>("view.contrast_min", cd.contrast_min);
      cd.contrast_min = contrast_min;

      float contrast_max = ptree.get<float>("view.contrast_max", cd.contrast_max);
      cd.contrast_max = contrast_max;

      // Special
      special_group_box->setHidden(ptree.get<bool>("special.hidden", false));

      bool image_ratio_enabled = ptree.get<bool>("special.image_ratio_enabled", cd.vibrometry_enabled);
      cd.vibrometry_enabled.exchange(image_ratio_enabled);

      int q_vibro = ptree.get<int>("special.image_ratio_q", cd.vibrometry_q);
      cd.vibrometry_q.exchange(q_vibro);

      bool average_enabled = ptree.get<bool>("special.average_enabled", is_enabled_average_);
      is_enabled_average_ = average_enabled;

      // Record
      record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

      // Import
      import_group_box->setHidden(ptree.get<bool>("import.hidden", false));
    }
  }

  void MainWindow::save_ini(const std::string& path)
  {
    boost::property_tree::ptree ptree;
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
    gui::GroupBox	*image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
    gui::GroupBox	*view_group_box = findChild<gui::GroupBox*>("View");
    gui::GroupBox	*special_group_box = findChild<gui::GroupBox*>("Vibrometry");
    gui::GroupBox	*record_group_box = findChild<gui::GroupBox*>("Record");
    gui::GroupBox	*import_group_box = findChild<gui::GroupBox*>("Import");

    // Image rendering
    ptree.put("image_rendering.hidden", image_rendering_group_box->isHidden());
    ptree.put("image_rendering.camera", camera_type_);
    ptree.put("image_rendering.frame_timeout", camera::FRAME_TIMEOUT);
    ptree.put("image_rendering.queue_size", q_max_size_);
    ptree.put("image_rendering.phase_number", cd.nsamples);
    ptree.put("image_rendering.p_index", cd.pindex);
    ptree.put("image_rendering.lambda", cd.lambda);
    ptree.put("image_rendering.z_distance", cd.zdistance);
    ptree.put("image_rendering.z_step", z_step_);
    ptree.put("image_rendering.algorithm", cd.algorithm);

    // View
    ptree.put("view.hidden", view_group_box->isHidden());
    ptree.put("view.view_mode", cd.view_mode);
    ptree.put("view.log_scale_enabled", cd.log_scale_enabled);
    ptree.put("view.shift_corners_enabled", cd.shift_corners_enabled);
    ptree.put("view.contrast_enabled", cd.contrast_enabled);
    ptree.put("view.contrast_min", cd.contrast_min);
    ptree.put("view.contrast_max", cd.contrast_max);

    // Special
    ptree.put("special.hidden", special_group_box->isHidden());
    ptree.put("special.image_ratio_enabled", cd.vibrometry_enabled);
    ptree.put("special.image_ratio_q", cd.vibrometry_q);
    ptree.put("special.average_enabled", is_enabled_average_);

    // Record
    ptree.put("record.hidden", record_group_box->isHidden());

    // Import
    ptree.put("import.hidden", import_group_box->isHidden());

    boost::property_tree::write_ini(path, ptree);
  }

  void MainWindow::split_string(const std::string& str, char delim, std::vector<std::string>& elts)
  {
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, delim))
      elts.push_back(item);
  }

  std::string MainWindow::format_batch_output(const std::string& path, unsigned int index)
  {
    std::string file_index;
    std::ostringstream convert;
    convert << std::setw(6) << std::setfill('0') << index;
    file_index = convert.str();

    std::vector<std::string> path_tokens;
    split_string(path, '.', path_tokens);

    return path_tokens[0] + "_" + file_index + "." + path_tokens[1];
  }
}