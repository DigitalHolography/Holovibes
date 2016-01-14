#include "main_window.hh"
#include "gui_gl_window.hh"
#include "gui_plot_window.hh"
#include "queue.hh"
#include "thread_recorder.hh"
#include "thread_csv_record.hh"
#include "compute_descriptor.hh"
#include "gpib_dll.hh"
#include "../GPIB/gpib_controller.hh"
#include "../GPIB/gpib_exceptions.hh"
#include "camera_exception.hh"
#include "config.hh"
#include "config.hh"
#include "info_manager.hh"

#define GLOBAL_INI_PATH "holovibes.ini"

namespace gui
{
  MainWindow::MainWindow(holovibes::Holovibes& holovibes, QWidget *parent)
    : QMainWindow(parent)
    , holovibes_(holovibes)
    , gl_window_(nullptr)
    , is_direct_mode_(true)
    , is_enabled_camera_(false)
    , is_enabled_average_(false)
    , is_batch_img_(true)
    , is_batch_interrupted_(false)
    , z_step_(0.1f)
    , camera_type_(holovibes::Holovibes::NONE)
    , last_contrast_type_("magnitude")
    , plot_window_(nullptr)
    , record_thread_(nullptr)
    , CSV_record_thread_(nullptr)
    , file_index_(1)
    , gpib_interface_(nullptr)
  {
    ui.setupUi(this);
    this->setWindowIcon(QIcon("icon1.ico"));
    InfoManager::get_manager(this->findChild<gui::GroupBox*>("Info"));

    camera_visible(false);
    record_visible(false);

    load_ini("holovibes.ini");
    layout_toggled(false);

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

    QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthModeComboBox");
    connect(depth_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(hide_endianess()));

    if (is_direct_mode_)
      global_visibility(false);

    // Display default values
    notify();
  }

  MainWindow::~MainWindow()
  {
    holovibes_.dispose_compute();
    holovibes_.dispose_capture();
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
    else if (cd.view_mode == holovibes::ComputeDescriptor::UNWRAPPED_ARGUMENT)
      view_mode->setCurrentIndex(3);
    else if (cd.view_mode == holovibes::ComputeDescriptor::UNWRAPPED_ARGUMENT_2)
      view_mode->setCurrentIndex(4);
    else if (cd.view_mode == holovibes::ComputeDescriptor::UNWRAPPED_ARGUMENT_3)
      view_mode->setCurrentIndex(5);
    else // Fallback on Modulus
      view_mode->setCurrentIndex(0);

    QSpinBox* unwrap_history_size = findChild<QSpinBox*>("unwrapSpinBox");
    unwrap_history_size->setValue(cd.unwrap_history_size);

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

  void MainWindow::layout_toggled(bool b)
  {
    unsigned int childCount = 0;
    std::vector<gui::GroupBox*> v;

    v.push_back(findChild<gui::GroupBox*>("ImageRendering"));
    v.push_back(findChild<gui::GroupBox*>("View"));
    v.push_back(findChild<gui::GroupBox*>("Vibrometry"));
    v.push_back(findChild<gui::GroupBox*>("Record"));
    v.push_back(findChild<gui::GroupBox*>("Import"));
    v.push_back(findChild<gui::GroupBox*>("Info"));

    for each (gui::GroupBox* var in v)
      childCount += !var->isHidden();

    if (childCount > 0)
      this->resize(QSize(childCount * 195, 385));
    else
      this->resize(QSize(195, 60));
  }

  void MainWindow::configure_holovibes()
  {
    open_file(holovibes_.get_launch_path() + "/" + GLOBAL_INI_PATH);
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
    gl_window_.reset(nullptr);
    if (!is_direct_mode_)
      holovibes_.dispose_compute();
    holovibes_.dispose_capture();
    camera_visible(false);
    record_visible(false);
    global_visibility(false);
  }

  void MainWindow::camera_adimec()
  {
    change_camera(holovibes::Holovibes::ADIMEC);
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
    display_info("Holovibes " + holovibes::version + "\n\n"
      "Scientists:\n"
      "Michael Atlan\n"
      "\n"
      "Developers:\n"
      "Eric Delanghe\n"
      "Arnaud Gaillard\n"
      "Geoffrey Le Gourriérec\n"
      "\n"
      "Jeffrey Bencteux\n"
      "Thomas Kostas\n"
      "Pierre Pagnoux\n"
      "\n"
      "Antoine Dillée\n"
      "Romain Cancillière\n");
  }

  void MainWindow::configure_camera()
  {
    open_file(boost::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
  }

  void MainWindow::set_image_mode(const bool value)
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

      if (gl_window_)
        gl_window_.reset(nullptr);

      // If direct mode
      if (value)
      {
        gl_window_.reset(new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_capture_queue()));
        is_direct_mode_ = true;

        global_visibility(false);
      }
      else
      {
        QCheckBox* pipeline_checkbox = findChild<QCheckBox*>("PipelineCheckBox");

        try
        {
          if (pipeline_checkbox->isChecked())
            holovibes_.init_compute(holovibes::ThreadCompute::PipeType::PIPELINE);
          else
            holovibes_.init_compute(holovibes::ThreadCompute::PipeType::PIPE);

          gl_window_.reset(new GuiGLWindow(pos, width, height, holovibes_, holovibes_.get_output_queue()));
          if (holovibes_.get_compute_desc().algorithm == holovibes::ComputeDescriptor::STFT)
          {
            GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");

            gl_widget->set_selection_mode(gui::eselection::STFT_ROI);
            connect(gl_widget, SIGNAL(stft_roi_zone_selected_update(holovibes::Rectangle)), this, SLOT(request_stft_roi_update(holovibes::Rectangle)),
              Qt::UniqueConnection);
            connect(gl_widget, SIGNAL(stft_roi_zone_selected_end()), this, SLOT(request_stft_roi_end()),
              Qt::UniqueConnection);
          }

          is_direct_mode_ = false;
          global_visibility(true);
        }
        catch (std::exception& e)
        {
          display_error(e.what());
        }
      }

      notify();
    }
  }

  void  MainWindow::set_phase_number(const int value)
  {
    if (!is_direct_mode_)
    {
      holovibes_.get_pipe()->request_update_n(value);
      notify();
    }
  }

  void  MainWindow::set_p(const int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < static_cast<int>(cd.nsamples))
      {
        // Synchronize with p_vibro
        QSpinBox* p_vibro = findChild<QSpinBox*>("pSpinBoxVibro");
        p_vibro->setValue(value);

        cd.pindex.exchange(value);
        holovibes_.get_pipe()->request_refresh();
      }
      else
        display_error("p param has to be between 0 and n");
    }
  }

  void MainWindow::increment_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex < cd.nsamples)
      {
        ++(cd.pindex);
        notify();
        holovibes_.get_pipe()->request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");
    }
  }

  void MainWindow::decrement_p()
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.pindex >= 0)
      {
        --(cd.pindex);
        notify();
        holovibes_.get_pipe()->request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");
    }
  }

  void  MainWindow::set_wavelength(const double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.lambda = static_cast<float>(value)* 1.0e-9f;
      holovibes_.get_pipe()->request_refresh();

      // Updating the GUI
      QLineEdit* boundary = findChild<QLineEdit*>("boundary");
      boundary->clear();
      boundary->insert(QString::number(holovibes_.get_boundary()));
    }
  }

  void  MainWindow::set_z(const double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.zdistance = static_cast<float>(value);
      holovibes_.get_pipe()->request_refresh();
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

  void MainWindow::set_z_step(const double value)
  {
    z_step_ = value;
    QDoubleSpinBox* z_spinbox = findChild<QDoubleSpinBox*>("zSpinBox");
    z_spinbox->setSingleStep(value);
  }

  void  MainWindow::set_algorithm(const QString value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      QSpinBox* phaseNumberSpinBox = findChild<QSpinBox*>("phaseNumberSpinBox");
      GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
      gl_widget->set_selection_mode(gui::eselection::ZOOM);

      cd.nsamples = 2;
      if (value == "1FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT1;
      else if (value == "2FFT")
        cd.algorithm = holovibes::ComputeDescriptor::FFT2;
      else if (value == "STFT")
      {
        cd.nsamples = 16;
        gl_widget->set_selection_mode(gui::eselection::STFT_ROI);
        connect(gl_widget, SIGNAL(stft_roi_zone_selected_update(holovibes::Rectangle)), this, SLOT(request_stft_roi_update(holovibes::Rectangle)),
          Qt::UniqueConnection);
        connect(gl_widget, SIGNAL(stft_roi_zone_selected_end()), this, SLOT(request_stft_roi_end()),
          Qt::UniqueConnection);
        cd.algorithm = holovibes::ComputeDescriptor::STFT;
      }
      else
        assert(!"Unknow Algorithm.");

      phaseNumberSpinBox->setValue(cd.nsamples);
      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_view_mode(const QString value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      // Reenabling phase number and p adjustments.
      QSpinBox* phase_number = findChild<QSpinBox*>("phaseNumberSpinBox");
      phase_number->setEnabled(true);

      QSpinBox* p = findChild<QSpinBox*>("pSpinBox");
      p->setEnabled(true);

      QCheckBox* pipeline_checkbox = findChild<QCheckBox*>("PipelineCheckBox");
      bool pipeline_checked = pipeline_checkbox->isChecked();

      std::cout << "Value = " << value.toUtf8().constData() << std::endl;

      if (value == "magnitude")
      {
        cd.view_mode = holovibes::ComputeDescriptor::MODULUS;
        last_contrast_type_ = value;
      }
      else if (value == "squared magnitude")
      {
        cd.view_mode = holovibes::ComputeDescriptor::SQUARED_MODULUS;
        last_contrast_type_ = value;
      }
      else if (value == "argument")
      {
        cd.view_mode = holovibes::ComputeDescriptor::ARGUMENT;
        last_contrast_type_ = value;
      }
      else
      {
        if (pipeline_checked)
        {
          // For now, phase unwrapping is only usable with the Pipe, not the Pipeline.
          display_error("Unwrapping is not available with the Pipeline.");
          QComboBox* contrast_type = findChild<QComboBox*>("viewModeComboBox");
          // last_contrast_type_ exists for this sole purpose...
          contrast_type->setCurrentIndex(contrast_type->findText(last_contrast_type_));
        }
        else
        {
          if (value == "phase 1")
            cd.view_mode = holovibes::ComputeDescriptor::UNWRAPPED_ARGUMENT;
          else if (value == "phase 2")
            cd.view_mode = holovibes::ComputeDescriptor::UNWRAPPED_ARGUMENT_2;
        }
      }

      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_unwrap_history_size(int value)
  {
    if (!is_direct_mode_)
    {
      holovibes_.get_compute_desc().unwrap_history_size = value;
      holovibes_.get_pipe()->request_update_unwrap_size(value);
    }
  }

  void MainWindow::set_unwrapping(const bool value)
  {
    if (!is_direct_mode_)
    {
      auto pipe = holovibes_.get_pipe();

      pipe->request_unwrapping(value);
      pipe->request_refresh();
    }
  }

  void MainWindow::set_autofocus_mode()
  {
    GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    gl_widget->set_selection_mode(gui::eselection::AUTOFOCUS);

    const float z_max = findChild<QDoubleSpinBox*>("zmaxDoubleSpinBox")->value();
    const float z_min = findChild<QDoubleSpinBox*>("zminDoubleSpinBox")->value();
    const unsigned int z_div = findChild<QSpinBox*>("zdivSpinBox")->value();
    const unsigned int z_iter = findChild<QSpinBox*>("ziterSpinBox")->value();
    holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

    if (desc.algorithm == holovibes::ComputeDescriptor::STFT)
    {
      display_error("You can't call autofocus in stft mode.");
      return;
    }

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

    desc.autofocus_zone = zone;
    holovibes_.get_pipe()->request_autofocus();
    gl_widget->set_selection_mode(gui::eselection::ZOOM);
  }

  void MainWindow::request_stft_roi_end()
  {
    holovibes_.get_pipe()->request_stft_roi_end();
  }

  void MainWindow::request_stft_roi_update(holovibes::Rectangle zone)
  {
    GLWidget* gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    holovibes::ComputeDescriptor& desc = holovibes_.get_compute_desc();

    desc.stft_roi_zone = zone;
    holovibes_.get_pipe()->request_stft_roi_update();
  }

  void MainWindow::request_autofocus_stop()
  {
    try
    {
      holovibes_.get_pipe()->request_autofocus_stop();
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
      holovibes_.get_compute_desc().contrast_enabled.exchange(value);

      set_contrast_min(contrast_min->value());
      set_contrast_max(contrast_max->value());

      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_auto_contrast()
  {
    if (!is_direct_mode_)
      holovibes_.get_pipe()->request_autocontrast();
  }

  void MainWindow::set_contrast_min(const double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.contrast_enabled)
      {
        if (cd.log_scale_enabled)
          cd.contrast_min = value;
        else
          cd.contrast_min = pow(10, value);

        holovibes_.get_pipe()->request_refresh();
      }
    }
  }

  void MainWindow::set_contrast_max(const double value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (cd.contrast_enabled)
      {
        if (cd.log_scale_enabled)
          cd.contrast_max = value;
        else
          cd.contrast_max = pow(10, value);

        holovibes_.get_pipe()->request_refresh();
      }
    }
  }

  void MainWindow::set_log_scale(const bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
      cd.log_scale_enabled.exchange(value);

      if (cd.contrast_enabled)
      {
        QDoubleSpinBox* contrast_min = findChild<QDoubleSpinBox*>("contrastMinDoubleSpinBox");
        QDoubleSpinBox* contrast_max = findChild<QDoubleSpinBox*>("contrastMaxDoubleSpinBox");
        set_contrast_min(contrast_min->value());
        set_contrast_max(contrast_max->value());
      }

      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_shifted_corners(const bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes_.get_compute_desc().shift_corners_enabled.exchange(value);
      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_vibro_mode(const bool value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      image_ratio_visible(value);
      cd.vibrometry_enabled.exchange(value);
      holovibes_.get_pipe()->request_refresh();
    }
  }

  void MainWindow::set_p_vibro(const int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < static_cast<int>(cd.nsamples) && value >= 0)
      {
        cd.pindex.exchange(value);
        notify();
        holovibes_.get_pipe()->request_refresh();
      }
      else
        display_error("p param has to be between 0 and n - 1");;
    }
  }

  void MainWindow::set_q_vibro(const int value)
  {
    if (!is_direct_mode_)
    {
      holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();

      if (value < static_cast<int>(cd.nsamples) && value >= 0)
      {
        holovibes_.get_compute_desc().vibrometry_q.exchange(value);
        holovibes_.get_pipe()->request_refresh();
      }
      else
        display_error("q param has to be between 0 and phase #");
    }
  }

  void MainWindow::set_average_mode(const bool value)
  {
    GLWidget * gl_widget = gl_window_->findChild<GLWidget*>("GLWidget");
    if (value)
      gl_widget->set_selection_mode(gui::eselection::AVERAGE);
    else
      gl_widget->set_selection_mode(gui::eselection::ZOOM);
    is_enabled_average_ = value;

    average_visible(value);
  }

  void MainWindow::set_average_graphic()
  {
    PlotWindow* plot_window = new PlotWindow(holovibes_.get_average_queue(), "ROI Average");

    connect(plot_window, SIGNAL(closed()), this, SLOT(dispose_average_graphic()), Qt::UniqueConnection);
    holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
    plot_window_.reset(plot_window);
  }

  void MainWindow::dispose_average_graphic()
  {
    plot_window_.reset(nullptr);
    if (!is_direct_mode_)
      holovibes_.get_pipe()->request_average_stop();
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
    const std::string path = path_line_edit->text().toUtf8();
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

    QSpinBox*  nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");

    int nb_of_frames = nb_of_frames_spinbox->value();
    std::string path = path_line_edit->text().toUtf8();
    holovibes::Queue* queue;

    try
    {
      if (float_output_checkbox->isChecked() && !is_direct_mode_)
      {
        std::shared_ptr<holovibes::ICompute> pipe = holovibes_.get_pipe();
        camera::FrameDescriptor frame_desc = holovibes_.get_output_queue().get_frame_desc();

        frame_desc.depth = sizeof(float);
        queue = new holovibes::Queue(frame_desc, global::global_config.float_queue_max_size, "FloatQueue");
        pipe->request_float_output(queue);
      }
      else if (is_direct_mode_)
        queue = &holovibes_.get_capture_queue();
      else
        queue = &holovibes_.get_output_queue();

      record_thread_.reset(new ThreadRecorder(
        *queue,
        path,
        nb_of_frames,
        this));

      connect(record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_image_record()));
      record_thread_->start();

      QPushButton* cancel_button = findChild<QPushButton*>("cancelPushButton");
      cancel_button->setDisabled(false);
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
    QCheckBox* float_output_checkbox = findChild<QCheckBox*>("RecordFloatOutputCheckBox");

    record_thread_.reset(nullptr);
    display_info("Record done");
    if (float_output_checkbox->isChecked() && !is_direct_mode_)
      holovibes_.get_pipe()->request_float_output_stop();
    if (!is_direct_mode_)
      global_visibility(true);
    record_but_cancel_visible(true);
  }

  void MainWindow::average_record()
  {
    if (plot_window_)
    {
      plot_window_->stop_drawing();
      plot_window_.reset(nullptr);
      holovibes_.get_pipe()->request_refresh();
    }

    QSpinBox* nb_of_frames_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
    nb_frames_ = nb_of_frames_spin_box->value();
    QLineEdit* output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    std::string output_path = output_line_edit->text().toUtf8();

    CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
      holovibes_.get_average_queue(),
      output_path,
      nb_frames_,
      this));
    connect(CSV_record_thread_.get(), SIGNAL(finished()), this, SLOT(finished_average_record()));
    CSV_record_thread_->start();

    global_visibility(false);
    record_but_cancel_visible(false);
    average_record_but_cancel_visible(false);
    QPushButton* roi_stop_push_button = findChild<QPushButton*>("ROIStopPushButton");
    roi_stop_push_button->setDisabled(false);
  }

  void MainWindow::finished_average_record()
  {
    CSV_record_thread_.reset(nullptr);
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
      plot_window_.reset(nullptr);
      holovibes_.get_pipe()->request_refresh();
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

    // Getting the path to the input batch file, and the number of frames to record.
    const std::string input_path = batch_input_line_edit->text().toUtf8();
    const unsigned int frame_nb = frame_nb_spin_box->value();

    try
    {
      // Only loading the dll at runtime
      gpib_interface_ = gpib::GpibDLL::load_gpib("gpib.dll", input_path);

      const std::string formatted_path = format_batch_output(path, file_index_);

      global_visibility(false);
      camera_visible(false);

      holovibes::Queue* q;

      if (is_direct_mode_)
        q = &holovibes_.get_capture_queue();
      else
        q = &holovibes_.get_output_queue();

      if (gpib_interface_->execute_next_block()) // More blocks to come, use batch_next_block method.
      {
        if (is_batch_img_)
        {
          record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, this));
          connect(record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_next_record()),
            Qt::UniqueConnection);
          record_thread_->start();
        }
        else
        {
          CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
            holovibes_.get_average_queue(),
            formatted_path,
            frame_nb,
            this));
          connect(CSV_record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_next_record()),
            Qt::UniqueConnection);
          CSV_record_thread_->start();
        }
      }
      else // There was only one block, so no need to record any further.
      {
        if (is_batch_img_)
        {
          record_thread_.reset(new ThreadRecorder(*q, formatted_path, frame_nb, this));
          connect(record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_finished_record()),
            Qt::UniqueConnection);
          record_thread_->start();
        }
        else
        {
          CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
            holovibes_.get_average_queue(),
            formatted_path,
            frame_nb,
            this));
          connect(CSV_record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_finished_record()),
            Qt::UniqueConnection);
          CSV_record_thread_->start();
        }
      }

      // Update the index to concatenate to the name of the next created file.
      ++file_index_;
    }
    catch (const std::exception& e)
    {
      display_error(e.what());
      batch_finished_record(false);
    }
  }

  void MainWindow::batch_next_record()
  {
    if (is_batch_interrupted_)
    {
      batch_finished_record(false);
      return;
    }

    record_thread_.reset(nullptr);

    QSpinBox * frame_nb_spin_box = findChild<QSpinBox*>("numberOfFramesSpinBox");
    std::string path;

    if (is_batch_img_)
      path = findChild<QLineEdit*>("pathLineEdit")->text().toUtf8();
    else
      path = findChild<QLineEdit*>("ROIOutputLineEdit")->text().toUtf8();

    holovibes::Queue* q;
    if (is_direct_mode_)
      q = &holovibes_.get_capture_queue();
    else
      q = &holovibes_.get_output_queue();

    std::string output_filename = format_batch_output(path, file_index_);
    const unsigned int frame_nb = frame_nb_spin_box->value();
    if (is_batch_img_)
    {
      try
      {
        if (gpib_interface_->execute_next_block())
        {
          record_thread_.reset(new ThreadRecorder(*q, output_filename, frame_nb, this));
          connect(record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_next_record()), Qt::UniqueConnection);
          record_thread_->start();
        }
        else
        {
          batch_finished_record(true);
        }
      }
      catch (const gpib::GpibInstrError& e)
      {
        display_error(e.what());
        batch_finished_record(false);
      }
    }
    else
    {
      try
      {
        if (gpib_interface_->execute_next_block())
        {
          CSV_record_thread_.reset(new ThreadCSVRecord(holovibes_,
            holovibes_.get_average_queue(),
            output_filename,
            frame_nb,
            this));
          connect(CSV_record_thread_.get(),
            SIGNAL(finished()),
            this,
            SLOT(batch_next_record()), Qt::UniqueConnection);
          CSV_record_thread_->start();
        }
        else
          batch_finished_record(true);
      }
      catch (const gpib::GpibInstrError& e)
      {
        display_error(e.what());
        batch_finished_record(false);
      }
    }

    // Update the index to concatenate to the name of the next created file.
    ++file_index_;
  }

  void MainWindow::batch_finished_record()
  {
    batch_finished_record(true);
  }

  void MainWindow::batch_finished_record(bool no_error)
  {
    record_thread_.reset(nullptr);
    CSV_record_thread_.reset(nullptr);
    gpib_interface_.reset();

    file_index_ = 1;
    if (!is_direct_mode_)
      global_visibility(true);
    camera_visible(true);
    if (no_error)
      display_info("Batch record done");

    if (plot_window_)
    {
      plot_window_->stop_drawing();
      holovibes_.get_pipe()->request_average(&holovibes_.get_average_queue());
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

    camera::FrameDescriptor frame_desc = {
      width_spinbox->value(),
      height_spinbox->value(),
      // 0:depth = 8, 1:depth = 16
      depth_spinbox->currentIndex() + 1,
      5.42f, // There's no way to find this...
      (big_endian_checkbox->currentText() == QString("Big Endian") ? camera::endianness::BIG_ENDIAN : camera::endianness::LITTLE_ENDIAN) };

    camera_visible(false);
    record_visible(false);
    global_visibility(false);
    gl_window_.reset(nullptr);
    holovibes_.dispose_compute();
    holovibes_.dispose_capture();
    holovibes_.init_import_mode(
      file_src,
      frame_desc,
      loop_checkbox->isChecked(),
      fps_spinbox->value(),
      start_spinbox->value(),
      end_spinbox->value(),
      global::global_config.input_queue_max_size);
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
    // Avoiding "unused variable" warning.
    static_cast<void>(event);
    save_ini("holovibes.ini");

    if (gl_window_)
      gl_window_->close();

    if (plot_window_)
      plot_window_->close();
  }

  void MainWindow::global_visibility(const bool value)
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

  void MainWindow::camera_visible(const bool value)
  {
    is_enabled_camera_ = value;
    gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("ImageRendering");
    image_rendering->setDisabled(!value);
    QAction* settings = findChild<QAction*>("actionSettings");
    settings->setDisabled(!value);
  }

  void MainWindow::contrast_visible(const bool value)
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

  void MainWindow::record_visible(const bool value)
  {
    gui::GroupBox* image_rendering = findChild<gui::GroupBox*>("Record");
    image_rendering->setDisabled(!value);
  }

  void MainWindow::record_but_cancel_visible(const bool value)
  {
    QLabel* nb_of_frames_label = findChild<QLabel*>("numberOfFramesLabel");
    nb_of_frames_label->setDisabled(!value);
    QSpinBox* nb_of_frames_spinbox = findChild<QSpinBox*>("numberOfFramesSpinBox");
    nb_of_frames_spinbox->setDisabled(!value);
    QToolButton* browse_button = findChild<QToolButton*>("ImageOutputBrowsePushButton");
    browse_button->setDisabled(!value);
    QLineEdit* path_line_edit = findChild<QLineEdit*>("pathLineEdit");
    path_line_edit->setDisabled(!value);
    QPushButton* record_button = findChild<QPushButton*>("recPushButton");
    record_button->setDisabled(!value);
  }

  void MainWindow::image_ratio_visible(const bool value)
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

  void MainWindow::average_visible(const bool value)
  {
    QLabel* roi_file_label = findChild<QLabel*>("ROIFileLabel");
    roi_file_label->setDisabled(!value);
    QToolButton* roi_browse_button = findChild<QToolButton*>("ROIFileBrowseToolButton");
    roi_browse_button->setDisabled(!value);
    QLineEdit* roi_file_line_edit = findChild<QLineEdit*>("ROIFileLineEdit");
    roi_file_line_edit->setDisabled(!value);
    QPushButton* save_roi_button = findChild<QPushButton*>("saveROIPushButton");
    save_roi_button->setDisabled(!value);
    QPushButton* load_roi_button = findChild<QPushButton*>("loadROIPushButton");
    load_roi_button->setDisabled(!value);
  }

  void MainWindow::average_record_but_cancel_visible(const bool value)
  {
    QLabel* roi_output_file_label = findChild<QLabel*>("ROIOuputLabel");
    roi_output_file_label->setDisabled(!value);
    QToolButton* roi_output_push_button = findChild<QToolButton*>("ROIOutputToolButton");
    roi_output_push_button->setDisabled(!value);
    QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("ROIOutputLineEdit");
    roi_output_line_edit->setDisabled(!value);
    QPushButton* roi_push_button = findChild<QPushButton*>("ROIPushButton");
    roi_push_button->setDisabled(!value);
  }

  void MainWindow::change_camera(const holovibes::Holovibes::camera_type camera_type)
  {
    if (camera_type != holovibes::Holovibes::NONE)
    {
      try
      {
        camera_visible(false);
        record_visible(false);
        global_visibility(false);
        gl_window_.reset(nullptr);
        holovibes_.dispose_compute();
        holovibes_.dispose_capture();
        holovibes_.init_capture(camera_type);
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

  void MainWindow::display_error(const std::string msg)
  {
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Critical);
    msg_box.exec();
  }

  void MainWindow::display_info(const std::string msg)
  {
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
  }

  void MainWindow::open_file(const std::string& path)
  {
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
  }

  void MainWindow::load_ini(const std::string& path)
  {
    boost::property_tree::ptree ptree;
    gui::GroupBox *image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
    gui::GroupBox *view_group_box = findChild<gui::GroupBox*>("View");
    gui::GroupBox *special_group_box = findChild<gui::GroupBox*>("Vibrometry");
    gui::GroupBox *record_group_box = findChild<gui::GroupBox*>("Record");
    gui::GroupBox *import_group_box = findChild<gui::GroupBox*>("Import");
    gui::GroupBox *info_group_box = findChild<gui::GroupBox*>("Info");

    QAction*      image_rendering_action = findChild<QAction*>("actionImage_rendering");
    QAction*      view_action = findChild<QAction*>("actionView");
    QAction*      special_action = findChild<QAction*>("actionSpecial");
    QAction*      record_action = findChild<QAction*>("actionRecord");
    QAction*      import_action = findChild<QAction*>("actionImport");
    QAction*      info_action = findChild<QAction*>("actionInfo");

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
      holovibes::Config& config = global::global_config;
      // Config
      config.input_queue_max_size = ptree.get<int>("config.input_queue_max_size", config.input_queue_max_size);
      config.output_queue_max_size = ptree.get<int>("config.output_queue_max_size", config.output_queue_max_size);
      config.float_queue_max_size = ptree.get<int>("config.float_queue_max_size", config.float_queue_max_size);
      config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
      config.flush_on_refresh = ptree.get<int>("config.flush_on_refresh", config.flush_on_refresh);
      config.reader_buf_max_size = ptree.get<int>("config.reader_buf_max_size", config.reader_buf_max_size);
      config.unwrap_history_size = ptree.get<int>("config.unwrap_history_size", config.unwrap_history_size);

      // Camera type
      const int camera_type = ptree.get<int>("image_rendering.camera", 0);
      change_camera((holovibes::Holovibes::camera_type)camera_type);

      // Image rendering
      image_rendering_action->setChecked(!ptree.get<bool>("image_rendering.hidden", false));
      image_rendering_group_box->setHidden(ptree.get<bool>("image_rendering.hidden", false));

      cd.nsamples = ptree.get<unsigned short>("image_rendering.phase_number", cd.nsamples);

      const unsigned short p_index = ptree.get<unsigned short>("image_rendering.p_index", cd.pindex);
      if (p_index >= 0 && p_index < cd.nsamples)
        cd.pindex.exchange(p_index);

      cd.lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);

      cd.zdistance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);

      const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
      if (z_step > 0.0f)
        z_step_ = z_step;

      cd.algorithm = static_cast<holovibes::ComputeDescriptor::fft_algorithm>(
        ptree.get<int>("image_rendering.algorithm", cd.algorithm));

      // View
      view_action->setChecked(!ptree.get<bool>("view.hidden", false));
      view_group_box->setHidden(ptree.get<bool>("view.hidden", false));

      cd.view_mode = static_cast<holovibes::ComputeDescriptor::complex_view_mode>(
        ptree.get<int>("view.view_mode", cd.view_mode));

      cd.unwrap_history_size = config.unwrap_history_size;

      cd.log_scale_enabled.exchange(
        ptree.get<bool>("view.log_scale_enabled", cd.log_scale_enabled));

      cd.shift_corners_enabled.exchange(
        ptree.get<bool>("view.shift_corners_enabled", cd.shift_corners_enabled));

      cd.contrast_enabled.exchange(
        ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled));

      cd.contrast_min = ptree.get<float>("view.contrast_min", cd.contrast_min);

      cd.contrast_max = ptree.get<float>("view.contrast_max", cd.contrast_max);

      // Special
      special_action->setChecked(!ptree.get<bool>("special.hidden", false));
      special_group_box->setHidden(ptree.get<bool>("special.hidden", false));

      cd.vibrometry_enabled.exchange(
        ptree.get<bool>("special.image_ratio_enabled", cd.vibrometry_enabled));

      cd.vibrometry_q.exchange(
        ptree.get<int>("special.image_ratio_q", cd.vibrometry_q));

      is_enabled_average_ = ptree.get<bool>("special.average_enabled", is_enabled_average_);

      // Record
      record_action->setChecked(!ptree.get<bool>("record.hidden", false));
      record_group_box->setHidden(ptree.get<bool>("record.hidden", false));

      // Import
      import_action->setChecked(!ptree.get<bool>("import.hidden", false));
      import_group_box->setHidden(ptree.get<bool>("import.hidden", false));

      // Info
      info_action->setChecked(!ptree.get<bool>("info.hidden", false));
      info_group_box->setHidden(ptree.get<bool>("info.hidden", false));

      // Autofocus
      cd.autofocus_size.exchange(ptree.get<int>("autofocus.size", cd.autofocus_size));
    }
  }

  void MainWindow::save_ini(const std::string& path)
  {
    boost::property_tree::ptree ptree;
    holovibes::ComputeDescriptor& cd = holovibes_.get_compute_desc();
    gui::GroupBox *image_rendering_group_box = findChild<gui::GroupBox*>("ImageRendering");
    gui::GroupBox *view_group_box = findChild<gui::GroupBox*>("View");
    gui::GroupBox *special_group_box = findChild<gui::GroupBox*>("Vibrometry");
    gui::GroupBox *record_group_box = findChild<gui::GroupBox*>("Record");
    gui::GroupBox *import_group_box = findChild<gui::GroupBox*>("Import");
    gui::GroupBox *info_group_box = findChild<gui::GroupBox*>("Info");
    holovibes::Config& config = global::global_config;

    // Config
    ptree.put("config.input_queue_max_size", config.input_queue_max_size);
    ptree.put("config.output_queue_max_size", config.output_queue_max_size);
    ptree.put("config.float_queue_max_size", config.float_queue_max_size);
    ptree.put("config.frame_timeout", config.frame_timeout);
    ptree.put("config.flush_on_refresh", config.flush_on_refresh);
    ptree.put("config.reader_buf_max_size", config.reader_buf_max_size);
    ptree.put("config.unwrap_history_size", config.unwrap_history_size);

    // Image rendering
    ptree.put("image_rendering.hidden", image_rendering_group_box->isHidden());
    ptree.put("image_rendering.camera", camera_type_);
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

    // Info
    ptree.put("info.hidden", info_group_box->isHidden());

    // Autofocus
    ptree.put("autofocus.size", cd.autofocus_size);

    boost::property_tree::write_ini(holovibes_.get_launch_path() + "/" + path, ptree);
  }

  void MainWindow::split_string(const std::string& str, const char delim, std::vector<std::string>& elts)
  {
    std::stringstream ss(str);
    std::string item;

    while (std::getline(ss, item, delim))
      elts.push_back(item);
  }

  std::string MainWindow::format_batch_output(const std::string& path, const unsigned int index)
  {
    std::string file_index;
    std::ostringstream convert;
    convert << std::setw(6) << std::setfill('0') << index;
    file_index = convert.str();

    std::vector<std::string> path_tokens;
    split_string(path, '.', path_tokens);

    return path_tokens[0] + "_" + file_index + "." + path_tokens[1];
  }

  void MainWindow::hide_endianess()
  {
    QComboBox* depth_cbox = findChild<QComboBox*>("ImportDepthModeComboBox");
    QString curr_value = depth_cbox->currentText();
    QComboBox* imp_cbox = findChild<QComboBox*>("ImportEndianModeComboBox");

    // Changing the endianess when depth = 8 makes no sense
    imp_cbox->setEnabled(curr_value != "8");
  }
}